import matplotlib.pyplot as plt
import numpy as np


# Data to plot
labels = 'Anthropogenic NMVOCs (126.9 Tg/yr)', 'Anthropogenic BC (4.8 Tg/yr)', 'Anthropogenic POA (10.5 Tg/yr)', 'Anthropogenic $SO2$ (55.2 Tg/yr)', 'Anthropogenic $NH3$ (41.6 Tg/yr)', 'Biomass Burning Aerosol (49.1 Tg/yr)'
sizes = [126.9, 4.8, 10.5, 55.2, 41.6, 49.1]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'orange', 'purple']


# Plot
f, ax = plt.subplots(figsize=(5,5))
ax.pie(sizes, colors=colors, shadow=True, startangle=140)
ax.set_title('Average Anthropogenic Emission in Year 2000')
plt.axis('equal')
plt.legend(labels, bbox_to_anchor=(1.1, .5), loc='center right')
plt.show()