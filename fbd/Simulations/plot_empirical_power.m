function plot_empirical_power(ax, ps_pos, ps_neg, ngrid)
% Empirical CDFs: R_alt(alpha)=P(p_pos<=alpha), R_nul(alpha)=P(p_neg<=alpha)
    if nargin < 4, ngrid = 2000; end

    axes(ax); cla(ax);
    hold(ax,'on');

    alpha = linspace(0,1,ngrid);
    R_alt = arrayfun(@(a) mean(ps_pos <= a), alpha);
    R_nul = arrayfun(@(a) mean(ps_neg <= a), alpha);

    plot(ax, alpha, alpha, 'LineWidth',3, 'Color',[0,0,0,0.5], 'LineStyle',':');
    plot(ax, alpha, R_nul, 'LineWidth',3, 'Color',[1,0,0,0.5]);
    plot(ax, alpha, R_alt, 'LineWidth',3, 'Color',[0,0,1,0.5]);

    xlabel(ax,'Significance cutoff $\alpha$');
    ylabel(ax,'Power');
    legend(ax, {'$p=\alpha$','Null','Alternative'}, 'Location','southeast');

    title(ax,'Empirical CDF Plot');
    axis(ax,'equal'); grid(ax,'on');

    ax.XTick = 0:0.1:1; ax.YTick = 0:0.1:1;
    xlim(ax,[0,1]); ylim(ax,[0,1]);

    set(ax,'FontName','Arial','FontSize',16,'LineWidth',1,'Box','off');
    ax.XLabel.FontSize = 15;
    ax.YLabel.FontSize = 15;
    ax.Title.FontSize  = 15;

    lgd = legend(ax); lgd.FontSize = 15;

    hold(ax,'off');
end
