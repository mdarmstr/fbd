%% Multiomics Dataset
% LEGEND FOR MAPPINGS
%
% Genotype: 0 -> WT
% Genotype: 1 -> GPR34-KO
% Genotype: 2 -> TREM2-KO
% Genotype: 3 -> GPR34/TREM2-KO
%
% Treatment 0 -> PBS
% Treatment 1 -> Myelin
%
% NOTE THAT ROW 20: Genotype is incorrectly indicated as ``4'' due to
% formatting error.

malign = readtable('meta_ST004205_AN006992.csv'); %#ok
malign = malign{:,{'Genotype','Treatment'}};
malign(19,1) = 1; %correction from the meta-data
disp(malign)

xalign = readtable('aligned_ST004205_MS1_MS2.csv');
