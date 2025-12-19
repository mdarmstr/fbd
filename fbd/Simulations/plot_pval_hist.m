function plot_pval_hist(ax, ps_pos, ps_neg, nbins)
% Plot overlaid histograms with shared bin edges
    if nargin < 4, nbins = 20; end

    axes(ax); cla(ax);
    hold(ax,'on'); box(ax,'on');

    all_p = [ps_pos(:); ps_neg(:)];
    if isempty(all_p) || all(~isfinite(all_p))
        hold(ax,'off'); return;
    end

    pmin = min(all_p); pmax = max(all_p);
    if pmin == pmax
        pmin = max(0, pmin-1e-3);
        pmax = min(1, pmax+1e-3);
    end
    edges = linspace(pmin, pmax, nbins);

    histogram(ax, ps_pos, edges, 'FaceColor','b', 'FaceAlpha',0.5);
    histogram(ax, ps_neg, edges, 'FaceColor','r', 'FaceAlpha',0.5);

    xlabel(ax,'p-value');
    ylabel(ax,'Frequency');
    legend(ax, {'Positive','Negative'}, 'Location','best','FontSize',15);

    ax.XLabel.FontSize = 15;
    ax.YLabel.FontSize = 15;
    ax.Title.FontSize  = 15;

    hold(ax,'off');
end
