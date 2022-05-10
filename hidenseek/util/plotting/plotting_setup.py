import matplotlib.pyplot as plt

font_family = 'Arial'
font_size = 8
title_size = 8
ticklabel_size = 8
subplot_labelsize = 14

plt.rcParams.update({'font.size': font_size,
                     'font.sans-serif' : font_family,
                     'font.family' : font_family,
                     'axes.titlesize' : title_size,
                     'axes.labelsize' : font_size,
                     'xtick.labelsize' : ticklabel_size,
                     'ytick.labelsize' : ticklabel_size})

A4_width = 8.27
A4_height = 11.69

# 174 mm in inches
cb_width = 174 / 25.4
narrow_cb_width = 114 / 25.4
# "Each figure should fit on a single 8.5” x 11” page"
cb_height = 11
