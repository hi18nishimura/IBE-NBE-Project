import numpy as np
import matplotlib.pyplot as plt

# 定数の設定
a = 8
p_max = 150

# 曲線描画用のデータ
p_in_curve = np.linspace(-p_max, p_max, 1000)
sign_p_in = np.where(p_in_curve >= 0, 1, -1)
p_out_curve = sign_p_in * 0.4 * (np.abs(p_in_curve) / p_max)**(1/a) + 0.5

# 特定のp_out値からp_inを逆算
p_out_targets = np.array([0.6, 0.7, 0.8, 0.85])
p_in_targets = p_max * ((p_out_targets - 0.5) / 0.4)**a

# グラフの設定
plt.figure(figsize=(12, 8), facecolor='white')
ax = plt.gca()

# フォントサイズ
TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 28, 24, 20, 20

# メイン曲線
plt.plot(p_out_curve, p_in_curve, color='#004c8c', linewidth=3, label=r'$p_{\mathrm{out}}$ vs $p_{\mathrm{in}}$')

# 特定の点をプロット
plt.scatter(p_out_targets, p_in_targets, color='#d32f2f', s=100, zorder=5)

# 点にラベルを付与
pos=[(10, -25), (10, 10), (10, 15), (10, 15)]
for x, y,pos in zip(p_out_targets, p_in_targets, pos):
    plt.annotate(f'({x}, {y:.2f})', (x, y), textcoords="offset points", xytext=pos, 
                 ha='right', fontsize=28, color='#b71c1c', fontweight='bold')

# 補助線と背景色
plt.axvline(0.1, color='#546e7a', linestyle='--', linewidth=1.5)
plt.axvline(0.9, color='#546e7a', linestyle='--', linewidth=1.5)
plt.axvspan(0.1, 0.25, color='#bbdefb', alpha=0.4)
plt.axvspan(0.75, 0.9, color='#bbdefb', alpha=0.4)

# 軸とタイトルの設定
plt.xlabel(r'$p_{\mathrm{out}}$', fontsize=LABEL_SIZE, labelpad=10)
plt.ylabel(r'$p_{\mathrm{in}}$', fontsize=LABEL_SIZE, labelpad=10)
plt.title(f'Relationship with Specific Points ($a={a}, p_{{max}}={p_max}$)', fontsize=TITLE_SIZE, pad=20)

plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
plt.xlim(0, 1)
plt.ylim(-p_max, p_max)
plt.grid(True, linestyle='--', alpha=0.3, color='#cfd8dc')

plt.legend(loc='upper left', frameon=True, fontsize=LEGEND_SIZE)
plt.tight_layout()
plt.savefig('power_normalize_plot.png',dpi=400)