%% 580 Dataset
% LEGEND FOR ORIGINAL MAPPINGS
% F_Diagnosis: 0 -> Breast Cancer | 1 -> Control
% F_Stage: 0 -> Stage 2 | 1 -> Stage 3 | 2 -> Stage 4 | 3 -> Stage 1 | 4 ->
% CNTRL
% AGE: Ordinal
%  

m580 = readtable('meta_ST000355_AN00580.csv'); %#ok
m580 = m580{:,{'F_Diagnosis','F_Stage','A_YOB'}};
%disp(m580)

%% 583 Dataset
% LEGEND FOR ORIGINAL MAPPINGS
% Diagnosis: 
%
% Diagnosis: 0 -> Control | 1 -> Breast Cancer
% Stage 0 -> Control | 1 -> Stage 2 | 2 -> Stage 3 
% Age: Ordinal

m583 = readtable('meta_ST000356_AN00583.csv'); %#ok
m583 = m583{:,{'Diagnosis','Stage','YOB'}};
%disp(m583)

%% Further data cleaning
% Map 580 to 583, and create new stages (3and4,1and2)

%Diagnosis
mnew = zeros(size(m580));

mnew(m580(:,1)==1,1) = 0;
mnew(m580(:,1)==0,1) = 1;

%Stage
mnew(m580(:,2)==4,2) = 0;
mnew(m580(:,2)==3 | m580(:,2) == 0,2) = 1; %Stages 1 and 2
mnew(m580(:,2)==1 | m580(:,2) == 2,2) = 2; %Stages 3 and 4

%% Saving everything to a .mat file for further analysis
%Saving the header files
h580 = readtable("ST000355_AN000580.csv").Properties.VariableNames;
h583 = readtable("ST000356_AN000583.csv").Properties.VariableNames;

T580 = readtable("ST000355_AN000580.csv");
T583 = readtable("ST000356_AN000583.csv");

X580 = T580{:,:};
X583 = T583{:,:};

D580 = mnew;
D583 = m583;

save('bcdata.mat','h580','h583','D580','D583','X580','X583')

