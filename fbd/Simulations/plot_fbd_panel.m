function plot_fbd_panel(axHist, axCdf, ps_pos, ps_neg)
    plot_pval_hist(axHist, ps_pos, ps_neg, 20);
    title(axHist, 'p-value histograms'); % this gets overwritten by column label if you want
    plot_empirical_power(axCdf, ps_pos, ps_neg, 2000);
end