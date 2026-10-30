"""Build the Task 1 PDF report (Chinese) from figures/ and results/.  python build_pdf.py <output.pdf>"""
import sys, json
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.utils import ImageReader

HERE = Path(__file__).resolve().parent; R = HERE / 'results'; F = HERE / 'figures'
pdfmetrics.registerFont(TTFont('CN', '/Library/Fonts/Arial Unicode.ttf'))
try:
    pdfmetrics.registerFont(TTFont('CNB', '/System/Library/Fonts/STHeiti Medium.ttc', subfontIndex=0))
except Exception:
    pdfmetrics.registerFont(TTFont('CNB', '/Library/Fonts/Arial Unicode.ttf'))
INK, MUTED, ACCENT, RULE = colors.HexColor('#1f1f1e'), colors.HexColor('#6b6a64'), colors.HexColor('#1baf7a'), colors.HexColor('#d9d8d2')
S = dict(
    title=ParagraphStyle('t', fontName='CNB', fontSize=20, leading=28, textColor=INK, spaceAfter=6),
    sub=ParagraphStyle('s', fontName='CN', fontSize=10.5, leading=16, textColor=MUTED, spaceAfter=14),
    h1=ParagraphStyle('h1', fontName='CNB', fontSize=14, leading=20, textColor=INK, spaceBefore=12, spaceAfter=6),
    h2=ParagraphStyle('h2', fontName='CNB', fontSize=11.5, leading=17, textColor=INK, spaceBefore=8, spaceAfter=4),
    body=ParagraphStyle('b', fontName='CN', fontSize=9.6, leading=15.5, textColor=INK, spaceAfter=4),
    bullet=ParagraphStyle('bl', fontName='CN', fontSize=9.6, leading=15.5, textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=2),
    box=ParagraphStyle('bx', fontName='CN', fontSize=10, leading=16.5, textColor=INK),
    cap=ParagraphStyle('c', fontName='CN', fontSize=8.5, leading=12.5, textColor=MUTED, spaceAfter=10),
    cell=ParagraphStyle('ce', fontName='CN', fontSize=8.3, leading=11.5, textColor=INK),
    cellb=ParagraphStyle('cb', fontName='CNB', fontSize=8.3, leading=11.5, textColor=INK),
)
W = A4[0] - 36 * mm


def P(t, s='body'): return Paragraph(t, S[s])
def B(items): return [Paragraph(t, S['bullet'], bulletText='•') for t in items]


def fig(name, caption, width=W):
    iw, ih = ImageReader(str(F / name)).getSize()
    return KeepTogether([Image(str(F / name), width=width, height=width * ih / iw), P(caption, 'cap')])


def table(rows, widths, bold_rows=(), highlight=()):
    data = [[Paragraph(str(c), S['cellb'] if i == 0 or i in bold_rows else S['cell']) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1)
    st = [('LINEBELOW', (0, 0), (-1, 0), .8, INK), ('LINEBELOW', (0, 1), (-1, -1), .3, RULE),
          ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]
    st += [('BACKGROUND', (0, i), (-1, i), colors.HexColor('#e8f6f0')) for i in highlight]
    t.setStyle(TableStyle(st)); return t


def boxed(text):
    t = Table([[Paragraph(text, S['box'])]], colWidths=[W])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f8f5')), ('LINEBEFORE', (0, 0), (0, -1), 3, ACCENT),
                           ('LEFTPADDING', (0, 0), (-1, -1), 10), ('TOPPADDING', (0, 0), (-1, -1), 8), ('BOTTOMPADDING', (0, 0), (-1, -1), 8)]))
    return t


def main(out):
    m = pd.read_csv(R / 'mean_metrics.csv').set_index(['window', 'model', 'target'])
    pr = pd.read_csv(R / 'paired_comparisons.csv').set_index(['window', 'target', 'new', 'baseline'])
    nz = pd.read_csv(R / 'noise_reference.csv').set_index(['window', 'target'])
    vs = json.load(open(R / 'valley_summary.json'))
    cells = [('EARLY', 'power'), ('EARLY', 'speed'), ('LATE', 'power'), ('LATE', 'speed')]
    names = [('TRAIN_MEDIAN', '训练中位数基线'), ('RAW', '不分带'), ('FIX5BW', '已有 5 带 Butterworth'), ('NAT_40', '已有固定 40 kHz'),
             ('NAT_55.5', '<b>固定 55.5 kHz</b>'), ('MEY2.5_55.5', '固定 55.5 kHz（窄过渡 ±2.5）'), ('EWT', '自适应 EWT')]
    main_rows = [['方法', 'EARLY 功率', 'EARLY 速度', 'LATE 功率', 'LATE 速度', '4 项平均']]
    for k, lab in names:
        v = [m.loc[(w, k, t), 'macro_MAPE'] for w, t in cells]
        main_rows.append([lab] + [f'{x:.2f}' for x in v] + [f'{sum(v) / 4:.2f}'])

    def prow(new, base, w, t, lab=''):
        r = pr.loc[(w, t, new, base)]
        return [lab, f'{w} {"功率" if t == "power" else "速度"}', f'{r.mean_delta_pp:+.2f}', f'[{r.ci95_low:.2f}, {r.ci95_high:.2f}]',
                f'{r.wilcoxon_p:.3f}', '是' if r.equivalent_pm1pp else '否']
    pair_rows = [['比较', '窗口·目标', '均值差', '95% 置信区间', 'Wilcoxon p', '±1 等效']]
    for new, base, lab in [('EWT', 'NAT_55.5', 'EWT − 固定 55.5'), ('NAT_55.5', 'NAT_40', '固定 55.5 − 已有 40 kHz'),
                           ('NAT_55.5', 'FIX5BW', '固定 55.5 − 已有 5 带'), ('NAT_55.5', 'RAW', '固定 55.5 − 不分带')]:
        for i, (w, t) in enumerate(cells):
            pair_rows.append(prow(new, base, w, t, lab if i == 0 else ''))

    st = []
    st += [P('ME5106 Mini Project 3 · 任务一', 'sub'), P('分带方法对照：固定 55.5 kHz、已有固定频带与自适应 EWT', 'title'),
           P('声学信号预测激光功率与扫描速度 · 2026 年 10 月', 'sub')]
    st.append(boxed('<b>问题</b>：EWT 方案的效果，来自"选对了频带"，还是来自"自适应调整"？<br/><br/>'
                    '<b>结论</b>：来自<b>选对频带</b>。EWT 的"自适应"在本数据上退化为常数：380 个训练子集中有 379 个估计出同一个约 55.5 kHz 的边界。'
                    '把边界固定在 55.5 kHz，结果与 EWT 统计等效（4 项平均误差 15.97% 对 15.88%），并明显优于已有的 40 kHz 两带和 5 带 Butterworth。'
                    '55.5 kHz 是测量链（传感器或传播路径）的固有谱谷：49 个样本中都存在，与功率、速度无关，激光关闭后仍在原位。'
                    '因此不需要每折重新估计边界，一次确定的固定滤波器即可。'))
    st.append(Spacer(1, 10))

    st.append(P('1　做了什么', 'h1'))
    st.append(P('在组内已冻结的流程上<b>只改变"怎么分带"</b>，其余全部保持不变，从而把"自适应""边界位置""滤波器形状"三个因素分开检验。'))
    st.append(table([['保持不变', '内容'],
                     ['信号', '8 kHz 四阶零相位高通；EARLY 窗口（起点后 2–10 ms）与 LATE 窗口（6–14 ms），各 1280 点'],
                     ['特征', '原始窗口和每个分量各 9 类：log RMS、能量占比、峭度、峰值因子、包络变异系数、包络 p95/中位数、谱质心、谱带宽、谱熵；同组 5 条轨迹取中位数'],
                     ['模型', '贪心 SFS（功率 ≤ 5 个、速度 ≤ 3 个特征，内层按工况划分）＋ SVR（功率 C=100, γ=0.05；速度 C=50, γ=0.1）'],
                     ['验证', '14 个工况逐一留出 × 3 个随机种子 × 2 个窗口 = 84 个外层折，复用冻结的划分']], [.14, .86]))
    st.append(Spacer(1, 6))
    st.append(table([['分带方式（共 48 种）', '含义', '用途'],
                     ['自适应 EWT', '每个训练子集重新估计边界', '被检验对象'],
                     ['<b>固定 55.5 kHz</b>', '边界固定，滤波器形状与 EWT 完全相同（过渡带 45.5–65.5 kHz）', '主对照：只去掉"自适应"'],
                     ['已有固定 40 kHz', '等宽两带（= 包里已有的 FIXEDK 对照）', '已有固定频带 ①'],
                     ['已有 5 带 Butterworth', '8–22–36–50–64–78 kHz，放进同一套 9 类特征', '已有固定频带 ②'],
                     ['不分带', '只用原始窗口', '分带本身有没有用'],
                     ['边界扫描 30–70 kHz', '每 2.5 kHz 一点，EWT 原生形状和恒定 ±2.5 kHz 过渡各一组', '位置效应（排除过渡宽度的干扰）'],
                     ['过渡带宽度', '边界 55.5 kHz，半宽 0.5 / 1 / 2.5 / 5 / 10 kHz', '滤波器形状效应'],
                     ['谷内微移', '54.5–56.5 kHz 共 8 个点', '边界抖动造成的"噪声"']], [.24, .5, .26]))
    st.append(P('统计方法：在 14 个工况上做配对比较（每个工况先对 3 个种子平均），报告均值差、95% bootstrap 置信区间（10000 次）和 Wilcoxon 符号秩检验；'
                '等效界限在实验前定为 ±1 个百分点（90% 置信区间完全落在界内才判为等效）。校验：原生 EWT 滤波器等于 d = γ·b 的 Meyer 滤波器（差 1e-14）；'
                '两带分解保留 100% 能量；复刻的 5 带滤波器与缓存一致（差 1e-15）；本机重跑的 EWT 与 Linux 存档最多差 0.12 个百分点。', 'cap'))

    st.append(PageBreak())
    st.append(P('2　结果', 'h1'))
    st.append(P('2.1　EWT 的"自适应"实际上是常数', 'h2'))
    st.append(fig('fig1_spectrum_filters.png', '图 1　(a) 49 个样本激光开启段的功率谱，55.5 kHz 处有稳定的谱谷；(b) 各分带方式的滤波器响应；'
                  '(c) EWT 在 380 个训练子集上估计的边界与各样本谱谷中心的分布。', W * .92))
    st.append(P('380 个训练子集中有 379 个只给出一个边界，范围 55.25–55.875 kHz（唯一例外多出一个 19.875 kHz 边界）。EWT 原生滤波器的过渡带横跨 45.5–65.5 kHz，'
                '边界在 55.25–55.875 kHz 之间移动对滤波器几乎没有影响。'))

    st.append(PageBreak())
    st.append(P('2.2　主对照：固定 55.5 kHz ≈ EWT，且明显优于已有固定频带', 'h2'))
    st.append(table(main_rows, [.3, .14, .14, .14, .14, .14], highlight=(5, 7)))
    st.append(P('工况宏平均 MAPE（%），3 个随机种子平均。EARLY 窗口的宏平均误差（物理单位）：固定 55.5 kHz 为功率 43.1 W、速度 81.6 mm/s；EWT 为 43.5 W、82.2 mm/s。', 'cap'))
    st.append(fig('fig3_main_comparison.png', '图 3　各方法的宏平均（上）与组平均（下）MAPE；点为 3 个种子平均，横线为种子间范围，虚线为训练中位数基线。'))
    st += B(['4 项平均下，不分带、已有 5 带、已有 40 kHz 都不如直接猜中位数（18.9–19.4% 对 18.73%），只有边界放在谷附近的方法明显更好。',
             '唯一偏向 EWT 的是 LATE 功率（好 0.66 个百分点）：小于同一模型换种子的波动（4.05），落在边界谷内微移的抖动范围内（1.61），且这一项所有方法都没赢过中位数基线，没有实际意义。'])

    st.append(PageBreak())
    st.append(P('2.3　配对检验（14 个留出工况）', 'h2'))
    st.append(table(pair_rows, [.25, .15, .12, .22, .13, .13]))
    st.append(P('负值表示前者更好；"±1 等效"指差值的 90% bootstrap 置信区间完全落在 ±1 个百分点以内。', 'cap'))
    st.append(fig('fig6_paired_conditions.png', '图 6　每个点为一个留出工况（3 个种子平均）；黑短线为均值，黑竖线为 95% 置信区间；绿色带为 ±1 个百分点等效区。', W * .9))

    st.append(PageBreak())
    st.append(P('2.4　边界位置：谱谷附近最稳，但不是调出来的最优值', 'h2'))
    st.append(fig('fig4_location_sweep.png', '图 4　边界位置从 30 到 70 kHz 扫描（两种滤波器形状），绿色带为实测谱谷范围。', W * .95))
    st += B(['EWT 原生形状下，4 项平均最好的 5 个边界全部在谷内（55.75、55.5、56.5、55、55.625 kHz，15.8–16.1%）；最差的是 50、65、67.5 kHz（19.5–19.9%）。',
             '过渡带固定为 ±2.5 kHz 时，最好的是 60、57.5、55.5 kHz（16.4–16.7%），40 kHz 为 18.7%，说明"谷附近更好"不是宽过渡带造成的假象。',
             '<b>曲线不平滑</b>：相邻 2.5 kHz 的边界可差 5–6 个百分点，单项上也有比 55.5 kHz 更好的位置（如 LATE 功率在 62.5 kHz）。这是 46 组样本上 SFS+SVR 的高方差；'
             '55.5 kHz 的依据是不用标签就能确定的物理谱谷，按测试结果挑边界属于过拟合。'])

    st.append(PageBreak())
    st.append(P('2.5　过渡带宽度与边界抖动', 'h2'))
    st.append(fig('fig5_robustness.png', '图 5　(a) 4 项平均随边界位置的变化；(b) 边界固定 55.5 kHz 时过渡带宽度的影响；(c) 谷内微移（54.5–56.5 kHz）带来的抖动与 EWT 的差别。'))
    noise_rows = [['', 'EARLY 功率', 'EARLY 速度', 'LATE 功率', 'LATE 速度'],
                  ['同一模型换种子的范围（中位数）'] + [f'{nz.loc[c, "median_seed_range_pp"]:.2f}' for c in cells],
                  ['边界谷内微移的标准差'] + [f'{nz.loc[c, "local_grid_sd_pp"]:.2f}' for c in cells],
                  ['边界谷内微移的范围'] + [f'{nz.loc[c, "local_grid_range_pp"]:.2f}' for c in cells],
                  ['|EWT − 固定 55.5|'] + [f'{abs(pr.loc[(w, t, "EWT", "NAT_55.5")].mean_delta_pp):.2f}' for w, t in cells]]
    st.append(table(noise_rows, [.36, .16, .16, .16, .16], highlight=(4,)))
    st.append(P('单位：百分点。', 'cap'))
    st += B(['过渡带半宽从 ±0.5 改到 ±10 kHz，4 项平均变化 ≤ 1.2、单项 ≤ 1.8 个百分点；原生宽过渡不比窄过渡差，固定版本沿用 EWT 形状即可。',
             'EARLY 中 86% 的外层折 EWT 边界恰为 55.5 kHz，约 45% 的预测与固定版本完全相同；其余差别来自内层子集边界的微小变化改变了 SFS 选第 2、3 个特征的路径。'
             '<b>自适应带来的不是更好的频带，而是特征选择路径的随机翻转。</b>'])

    st.append(PageBreak())
    st.append(P('2.6　谱谷的物理来源', 'h2'))
    st.append(fig('fig2_valley_invariance.png', '图 2　(a) 每个样本的谱谷中心（实心为激光开启段，空心为激光关闭后的衰减尾巴）；(b) 打印板上的分布；(c) 开启段、尾巴与空闲噪声的平均谱。'))
    pl, tl = vs['plateau'], vs['tail']; sp = vs['spearman_plateau_valley']
    st.append(table([['', '谷中心 [kHz]', '谷深 [dB]'],
                     ['激光开启段', f'{pl["mean_khz"]:.2f} ± {pl["sd_khz"]:.2f}（{pl["min_khz"]:.1f}–{pl["max_khz"]:.1f}）', f'中位数 {pl["depth_db_median"]:.1f}（最小 {pl["depth_db_min"]:.1f}）'],
                     ['激光关闭后的衰减尾巴', f'{tl["mean_khz"]:.2f} ± {tl["sd_khz"]:.2f}（{tl["min_khz"]:.1f}–{tl["max_khz"]:.1f}）', f'中位数 {tl["depth_db_median"]:.1f}（最小 {tl["depth_db_min"]:.1f}）'],
                     ['空闲噪声', '55.5 kHz 附近没有谷（50–60 kHz 单调上升）', '—']], [.3, .4, .3]))
    st.append(Spacer(1, 6))
    st += B([f'与工艺参数无关：与功率 ρ = {sp["power"]["rho"]:.2f}（p = {sp["power"]["p"]:.2f}），与速度 ρ = {sp["speed"]["rho"]:.2f}（p = {sp["speed"]["p"]:.2f}）。',
             '激光关闭后仍在原位（尾巴 − 开启段 = +0.02 ± 0.45 kHz）：此时熔池已不再发声，说明谷来自传播路径或传感器的传递函数（如反共振，或光程差约 3 mm 的两路干涉，c/2Δ ≈ 55.5 kHz）。',
             '空闲噪声中没有：说明它是声学通道的响应，而不是电子噪声（空闲谱中 48、56 kHz 尖峰为 8 kHz 整数倍电气谐波）。',
             f'与打印顺序弱相关（ρ = {sp["print_order"]["rho"]:.2f}），总漂移约 1–2 kHz，可能来自传感器温度变化，但比 ±10 kHz 的过渡带小一个数量级，对分带无影响。',
             '3 个未知样本的谷在 55.38（#41）、56.23（#43）、55.15 kHz（#49），固定滤波器同样适用。'])
    st.append(boxed('<b>物理结论</b>：55.5 kHz 是测量系统的固有频率特征，不属于熔池物理，也不随工艺参数变化。对固定的测量系统，它是常数，'
                    '用固定边界最合适；只有更换传感器或调整安装位置时才需要重新标定一次。'))

    st.append(PageBreak())
    st.append(P('2.7　机理：为什么谷附近分带更好', 'h2'))
    st.append(fig('fig8_feature_selection.png', '图 8　EARLY 窗口下各分带方式的选特征频率（42 次外层拟合中被选中的比例 %）。mode1 = 高频分量，mode2 = 低频分量。', W * .95))
    st += B(['功率：40 kHz 和谷附近分带的第一个特征都是低频带峰值因子 mode2_crest（被选 41–42 次）；差别在后续特征——谷附近会补上低频带带宽 / 质心和谱熵。',
             '速度：谷附近分带时主要依赖低频带能量占比（67%）和高频带包络 p95/中位数；40 kHz 和不分带时退回谱质心（83%、88%）。',
             '5 带 Butterworth 的 50–64 kHz 带把谷切在中间，功率只能依赖第 5 带谱熵和原始峭度，EARLY 功率误差高达 27%。'])
    st.append(fig('fig7_separability.png', '图 7　不训练模型的单特征区分度 ω² 随边界位置的变化（EARLY，描述性；高功率工况都在板子边缘，ω² 中混有位置效应）。', W * .95))
    st += B(['给定速度后的功率：高频带峭度在 50–56 kHz 一带 ω² 最高（53 kHz 处 0.933，55.5 kHz 处 0.918，40 kHz 处 0.859，原始峭度 0.82）。',
             '速度：低频带能量占比在 55.5 kHz 处有局部峰（0.264，40 kHz 为 0.194），但不是全局最高（68 kHz 为 0.288，原始 log RMS 为 0.30），只能说明谷是有效的分界点。'])

    st.append(PageBreak())
    st.append(P('2.8　对未知样本的影响', 'h2'))
    u = pd.read_csv(R / 'unknown_summary.csv').set_index(['sample_id', 'target', 'model'])['mean']
    urows = [['样本', '目标', '自适应 EWT', '固定 55.5 kHz', '窄过渡 55.5', '已有 40 kHz', '已有 5 带']]
    for s_ in [41, 43, 49]:
        for t, lab in [('power', '功率 [W]'), ('speed', '速度 [mm/s]' if s_ != 49 else '速度（已知 850）')]:
            urows.append([f'#{s_}' if t == 'power' else '', lab] + [f'{u.loc[(s_, t, k)]:.0f}' if t == 'speed' else f'{u.loc[(s_, t, k)]:.1f}'
                                                                     for k in ['EWT', 'NAT_55.5', 'MEY2.5_55.5', 'NAT_40', 'FIX5BW']])
    st.append(table(urows, [.08, .2, .144, .144, .144, .144, .144]))
    st.append(P('最终协议：46 组全部训练，内层 KFold(5) 按工况划分，3 个种子平均。', 'cap'))
    st += B(['全部 46 组的参考边界恰为 55.5 kHz，固定版本的<b>功率预测与 EWT 完全一致，提交的数字不用改</b>；速度相差 ≤ 25 mm/s（内层选特征路径不同）。',
             '换成 40 kHz 或 5 带，#41、#43 的功率预测低约 70–105 W：最终答案对分带方式敏感，分带必须事先按物理依据确定，不能看结果再挑。'])

    st.append(P('3　结论与建议', 'h1'))
    st += B(['<b>效果来自选对频带，不需要自适应</b>：EWT 每次都估计出同一个边界；固定 55.5 kHz 与 EWT 在 3/4 项上统计等效，4 项平均 15.97% 对 15.88%。',
             '<b>频带位置很关键</b>：谷附近分带比已有 40 kHz（EARLY 功率 −5.5 个百分点）和已有 5 带（−11.0）明显更好。',
             '<b>55.5 kHz 有物理依据</b>：测量链的固有谱谷，与工艺参数无关，激光关闭后依然存在。',
             '<b>建议</b>：报告中改为"固定 55.5 kHz 两带分解（EWT 型滤波器）"，说明"边界取自训练谱上稳定的谱谷，该谷经验证为测量系统特征"；EWT 作为一次性确定边界的工具介绍。',
             '过渡带宽窄影响不大（≤ 约 2 个百分点），沿用 EWT 原生形状即可。'])
    st.append(P('4　局限', 'h1'))
    st += B(['不是独立验证：数据已被多轮分析用过，边界扫描在看过第一版结果后设计，属于事后敏感性分析。',
             '55.5 kHz 由 EWT 在全部 46 组训练谱上（不用标签）确定，属于"设计好的固定滤波器"；但谷在每个样本上都存在，换任何训练子集结果几乎相同（379/380）。',
             '边界位置的结果噪声很大，单项上存在比 55.5 kHz 更好的位置，不能作为选边界的依据。',
             '所有方法按组平均都仍不如中位数基线（固定 55.5 kHz 为 14.8% / 16.3%，基线 12.1% / 12.1%），分带方式解决不了；根本原因在于固定截取 2–10 ms 窗口丢掉了扫描时长（t = L/v）信息，以及板上位置对幅值的影响。',
             'LATE 窗口（6–14 ms）对 850 mm/s 的轨迹（激光开启约 12–15 ms）可能已进入衰减段，解读需谨慎。',
             '谱谷成因（反共振或两路干涉）缺少传感器规格和安装几何佐证，只能确定它属于测量链而不是熔池。'])
    st.append(P('5　与任务二、任务三的分工', 'h1'))
    st += B(['谱谷物理分析（图 2）是选 55.5 kHz 的依据，归任务一；任务二需要时引用。',
             '特征区分度（图 7）与任务二"特征解释"有交叉，在此仅用于说明谷附近分带为何更好；各特征与功率、速度的关系以任务二为准。',
             '选特征频率（图 8）与任务三"模型依赖哪些信息"有关，但只是描述，不能代替消融，结论以任务三为准。',
             '如果组里改用固定 55.5 kHz，消融最好在固定版本上做；两者结果几乎一样，已在 EWT 上完成的不必重做，注明即可。'])
    st.append(P('附：文件与复现', 'h1'))
    st.append(P('代码位于 <font face="CN">project3/analysis/fixed55_control/</font>，依赖同 requirements-main.txt。依次运行 compare.py（约 4 分钟）、valley.py、separability.py、'
                'finalize_unknown.py、figures.py、build_pdf.py。全部指标和中间数据见交付包"数据"文件夹及其中的数据说明。'))

    def deco(c, d):
        c.saveState(); c.setFont('CN', 7.5); c.setFillColor(MUTED)
        c.drawString(18 * mm, 10 * mm, 'ME5106 Mini Project 3 · 任务一 分带方法对照'); c.drawRightString(A4[0] - 18 * mm, 10 * mm, str(d.page)); c.restoreState()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                            title='任务一 分带方法对照', author='ME5106 Mini Project 3')
    doc.build(st, onFirstPage=deco, onLaterPages=deco); print('saved', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / '任务一_分带方法对照报告.pdf'))
