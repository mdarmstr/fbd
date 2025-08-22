%CDF_plot
close all;

set(groot,'defaultTextInterpreter','latex');        % text objects
set(groot,'defaultLegendInterpreter','latex');      % legend
set(groot,'defaultAxesTickLabelInterpreter','latex'); % tick labels

figure('Units','inches','Position',[1 1 7 6]);  % larger figure size
hold on;
alpha = linspace(0,1,200);

R_alt  = arrayfun(@(a) mean(pVals_Positive  <= a), alpha);
R_nul  = arrayfun(@(a) mean(pVals_Negative  <= a), alpha);
plot(alpha, alpha, 'LineWidth', 3, 'Color', [0,0,0,0.5],'LineStyle',':'); % y = x reference line
plot(alpha, R_nul, 'LineWidth', 3, 'Color', [0 0 1 0.5]);
plot(alpha, R_alt, 'LineWidth', 3, 'Color', [1 0 0 0.5]);

xlabel('Significance cutoff $\alpha$');
ylabel('Proportion rejected');
legend('$p$ = $\alpha$','Null (Uniform)', 'Alternative', 'Location', 'southeast');
title('Empirical cumulative distributions of \textit{p}-values');

axis equal 
grid on;
ax = gca;
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.GridAlpha = 0.3;       % transparency of grid lines
ax.MinorGridLineStyle = '-';
%ax.XMinorGrid = 'on';
%ax.YMinorGrid = 'on';
ax.XTick = 0:0.1:1;
ax.YTick = 0:0.1:1;

xlim([0,1])
ylim([0,1])

set(gca, 'FontName', 'Arial', ...     % clean font
         'FontSize', 16, ...          % larger tick labels
         'LineWidth', 1, ...        % thicker axis lines
         'Box', 'off');               % remove top/right box lines

% Enlarge labels & title
ax = gca;
ax.XLabel.FontSize = 15;
ax.YLabel.FontSize = 15;
ax.Title.FontSize  = 15;

% Enlarge legend
lgd = legend;
lgd.FontSize = 14;
lgd.Box = 'off';

exportgraphics(gcf,'rejection_curve.pdf','ContentType','vector');