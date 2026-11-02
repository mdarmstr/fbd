% Analysis of breast cancer data

clear all; clc; close all;

load('bcdata.mat')

[X580p, x850m, x850s] = preprocess2D(X580,'Preprocessing',2);
[X583p, x853m, x853s] = preprocess2D(X583,'Preprocessing',2);

[~, parglmo11] = parglm(X580p,D580(:,2),'Preprocessing',0, 'Permutations', 1);
[~, parglmo22] = parglm(X583p,D583(:,2),'Preprocessing',0, 'Permutations', 1);

mdl12 = fbd(parglmo11,parglmo22,2); %independence testing not used here
mdl12.opt();
mdl12.pred_X1X2();

mdl21 = fbd(parglmo22,parglmo11,2);
mdl21.opt()
mdl21.pred_X1X2()

%reconstruction
Xp12 = mdl12.X1X2e;%.*x853s + x853m;
Xp21 = mdl21.X1X2e;%.*x850s + x850m;

[TX850, parglmo11] = parglm(X580p,D580(:,2),'Preprocessing',0, 'Permutations', 10000);
[TX853, parglmo22] = parglm(X583p,D583(:,2),'Preprocessing',0, 'Permutations', 10000);
[TXp12, parglmo12] = parglm(Xp12,D580(:,2),'Preprocessing',0, 'Permutations', 10000);
[TXp21, parglmo21] = parglm(Xp21,D583(:,2),'Preprocessing',0, 'Permutations', 10000);

levels = unique([D580(:,2); D583(:,2)]);
colours = lines(numel(levels));

plot_asca(parglmo11, D580(:,2), levels, colours, 'X11');
plot_asca(parglmo12, D580(:,2), levels, colours, 'X12');
plot_asca(parglmo21, D583(:,2), levels, colours, 'X21');
plot_asca(parglmo22, D583(:,2), levels, colours, 'X22');

export_glm_latex( ...
    {TX850, TXp12, TXp21, TX853}, ...
    {'X850', 'X12', 'X21', 'X853'}, ...
    'breast_cancer_ANOVA.tex');

function plot_asca(parglmo, labels, levels, colours, name)

asca_model = asca(parglmo);
scores = asca_model.factors{1}.scoresV(:,1:2);

fig = figure('Color','w','Units','centimeters', ...
    'Position',[3 3 9 9]);

ax = axes(fig);
hold(ax,'on');

for k = 1:numel(levels)
    rows = labels == levels(k);
    plot(ax, scores(rows,1), scores(rows,2), 'o', ...
        'Color',colours(k,:), 'MarkerFaceColor' ,colours(k,:), ...
        'MarkerSize',4,'LineWidth',0.7, ...
        'DisplayName',sprintf('Level %g',levels(k)));
end

title(ax, [name ' ASCA Scores']);
xlabel(ax, 'ASCA component 1','FontSize',9);
ylabel(ax, 'ASCA component 2','FontSize',9);

set(ax,'FontName','Arial','FontSize',8, ...
    'LineWidth',0.7,'TickDir','out','TickLength',[0.015 0.015]);

axis(ax,'square');
box(ax,'on');

legend(ax,'Location','northeast','FontSize',6,'Box','off');

set(fig,'PaperUnits','centimeters', ...
    'PaperSize',[9 9],'PaperPosition',[0 0 9 9]);

print(fig, ['ASCA_' name '.pdf'], '-dpdf');
print(fig, ['ASCA_' name '.png'], '-dpng','-r600');

end

function export_glm_latex(tables,names,filename) %% AI GENERATED

fid = fopen(filename,'w');
assert(fid ~= -1,'Could not open output file.');

% Preserve row labels if the tables use RowNames.
for k = 1:numel(tables)
    if ~isempty(tables{k}.Properties.RowNames)
        tables{k} = addvars(tables{k}, ...
            string(tables{k}.Properties.RowNames), ...
            'Before',1,'NewVariableNames','Source');
    end
end

headers = [{'Matrix'},tables{1}.Properties.VariableNames];
headers = cellfun(@latex_escape,headers,'UniformOutput',false);

fprintf(fid,'%s\n', ...
    '\begin{table}[htbp]', ...
    '\centering', ...
    '\small', ...
    '\caption{GLM results for the observed and reconstructed matrices.}', ...
    '\label{tab:breast-cancer-anova}', ...
    ['\begin{tabular}{ll',repmat('r',1,width(tables{1})-1),'}'], ...
    '\hline', ...
    [strjoin(headers,' & '),' \\'], ...
    '\hline');

for k = 1:numel(tables)
    T = tables{k};

    if k > 1
        fprintf(fid,'%s\n','\hline\hline');
    end

    for i = 1:height(T)
        cells = cell(1,width(T)+1);
        cells{1} = '';

        % Matrix label appears at the top left of each block.
        if i == 1
            cells{1} = ['\textbf{',latex_escape(names{k}),'}'];
        end

        for j = 1:width(T)
            value = T{i,j};
            if iscell(value), value = value{1}; end

            if isnumeric(value) || islogical(value)
                cells{j+1} = sprintf('%.4g',value);
            else
                cells{j+1} = latex_escape(char(string(value)));
            end
        end

        fprintf(fid,'%s\n',[strjoin(cells,' & '),' \\']);
    end
end

fprintf(fid,'%s\n', ...
    '\hline', ...
    '\end{tabular}', ...
    '\end{table}');

fclose(fid);
end

function text = latex_escape(text)
text = char(text);
text = strrep(text,'_','\_');
text = strrep(text,'%','\%');
text = strrep(text,'&','\&');
text = strrep(text,'#','\#');
end