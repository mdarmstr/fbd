%% Multiomics Dataset
% LEGEND FOR MAPPINGS
% O1...O18: 1-17
% Me1...Me36: 18-41
%
% T0: 0
% T1: 1
% T2: 2
%
% Omaluzimab: 0
% Mepolizumab: 1
%
% Female: 0
% Male: 1

malign = readtable('meta_ST002948_AN004834.csv'); %#ok
malign = malign{:,{'SubjectID','Time','Treatment','Gender'}};
disp(malign)

% Row block samples
blk1_smpl = 1:2:max(malign(:,1));
disp(blk1_smpl')

blk2_smpl = 2:2:max(malign(:,1));
disp(blk2_smpl')

%Logical mask
blk1_msk = ismember(malign(:,1),blk1_smpl);
blk2_msk = ismember(malign(:,1),blk2_smpl);

blk1_mta = malign(blk1_msk,:);
blk2_mta = malign(blk2_msk,:);

aligned_data = readtable('aligned_ST002948_HILIC_REVER.csv'); %#ok
%colNum = find(strcmp(aligned_data.Properties.VariableNames,
%'REVER_Sphingosine')) %Find first reverse phase analysis
%colNum = 21

data_mat = aligned_data{:,2:end};

rowblk1 = data_mat(blk1_msk,:);
rowblk2 = data_mat(blk2_msk,:);

trainblk11 = rowblk1(:,1:20);
trainblk22 = rowblk2(:,21:end);

testblk12 = rowblk1(:,21:end);
testblk21 = rowblk2(:,1:20);

disp('blk1 unique levels')
disp(unique(blk1_mta(:,2:4), 'rows', 'stable'))

disp('blk2 unique levels')
disp(unique(blk2_mta(:,2:4), 'rows', 'stable'))

% basic size checks
disp(size(malign,1))
disp(size(data_mat,1))
assert(size(malign,1) == size(data_mat,1), 'Metadata and data rows do not match.');

% subject split based on stable observed order, not odd/even numeric ID
subs = unique(malign(:,1), 'stable');
blk1_smpl = subs(1:2:end);
blk2_smpl = subs(2:2:end);

blk1_msk = ismember(malign(:,1), blk1_smpl);
blk2_msk = ismember(malign(:,1), blk2_smpl);

blk1_mta = malign(blk1_msk,:);
blk2_mta = malign(blk2_msk,:);

rowblk1 = data_mat(blk1_msk,:);
rowblk2 = data_mat(blk2_msk,:);

% check no subject leakage
assert(isempty(intersect(unique(blk1_mta(:,1)), unique(blk2_mta(:,1)))), ...
    'Subjects appear in both blocks.');

% check design balance
disp('blk1 unique design rows:')
disp(unique(blk1_mta(:,2:4), 'rows'))

disp('blk2 unique design rows:')
disp(unique(blk2_mta(:,2:4), 'rows'))

% check counts by design row
[G1,~,g1] = unique(blk1_mta(:,2:4), 'rows');
[G2,~,g2] = unique(blk2_mta(:,2:4), 'rows');

disp('blk1 design counts:')
disp([G1 accumarray(g1,1)])

disp('blk2 design counts:')
disp([G2 accumarray(g2,1)])

% check expected variable counts
assert(size(data_mat,2) == 41, 'Unexpected variable count.');

save('ST002948.mat','trainblk11','trainblk22','testblk12','testblk21','blk1_mta','blk2_mta')
