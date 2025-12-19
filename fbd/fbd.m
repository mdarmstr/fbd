classdef fbd < handle

%% Fusion by Design (FBD)
% A family of statistical tests and linear functions that align X1 to X2
% and evaluate their statistical independence. with no common sample or 
% feature modality via a common experimental design in F1, F2.
%
% IMPORTANT NOTE: Test all factors for evidence of significance across
% parglmoA and parglmoB. Reinitialize with only the factors of interest in
% F1, F2.
%
% INPUTS
% parglmoA - output of parglm.m in MEDA toolbox for X1, F1
% parglmoB - output of parglm.m in MEDA toolbox for X2, F2
% OUTPUTS
% -OBJ
%   .p         - p-value for NNSPM heteromodal statistical test
%   .T1oe      - Original PCA scores for X1 + E1, with respect to fctrs
%   .T1r       - Rotated PCA scores for X1 + E1, with respect to fctrs, X2, F2
%   .T2oe      - Original PCA scores for X2 + E2, with respect to fctrs
%   .R         - Opimtal rotation for X_1 to X_2    
%   .X1X2e     - Predicted off-diagonal entries with uncertainty    
%   .X1X2n     - Expected values for off-diagonal entries    
%   .F1_sorted - Sorted values of F1 in ascending order    
%   .F2_sorted - Sorted values of F2 in ascending order    
%   .X1_sorted - X1 sorted with respect to F1    
%   .X2_sorted - X2 sorted with respect to F2    
%   .idx       - Index used to sort X1, F1.    
%
% Software preparation:  Install MEDA-Toolbox following readme file;
%
% coded by: Michael Sorochan Armstrong (mdarmstr@ugr.es)
%           Jose Camacho Paez (josecamacho@ugr.es)
%
% last modification: DEC 2025
%
% Copyright (C) 2025  University of Granada, Granada
% Copyright (C) 2025  Michael Sorochan Armstrong, Jose Camacho Paez
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
% MEDA dependencies: parglmo

    properties
        n_perms       = []
        mdl1          = []
        mdl2          = []
        p             = []
        T1oe          = []
        T1r           = []
        T2oe          = []
        R             = []
        X1X2e         = []
        X1X2n         = []
        F1_sorted     = []
        F2_sorted     = []
        X1_sorted     = []
        X2_sorted     = []
        idx           = []
    end

    methods
        function obj = fbd(parglmoA, parglmoB, n_perms)
            % --- Required arguments ---
            if nargin < 5
                n_perms = 1000;
            elseif nargin < 4
                error('FBD requires X1, F1, X2, and F2.');
            end

            % --- Add paths (helpers + MEDA) ---
            addpath(genpath('.'));
            addpath(genpath('../MEDA'));
            if ~license('test', 'Statistics_Toolbox')
                error('Statistics and Machine Learning Toolbox not available.');
            end

            % --- Store data ---
            obj.mdl1 = parglmoA;
            obj.mdl2 = parglmoB;
            obj.n_perms = n_perms;
        end

        function test(obj)

            [obj.p,obj.T1oe,obj.T1r,obj.T2oe,obj.R] = nnspt(obj.mdl1,obj.mdl2,obj.n_perms);
            
        end

        function pred_X1X2(obj)
            %(X_1n + E_1)P_1O_1>2P_2^T            
            %For simplicity, we are using the whole design matrix, assuming
            %the user has correctly trimmed the model previously.
            
            % Calculate matrix of expected values - first order
            % experimental designs consistently.
            X1 = obj.mdl1.data;
            X2 = obj.mdl2.data;

            F1 = obj.mdl1.design;
            F2 = obj.mdl2.design;

            [F1,idx1] = sort(F1,"ascend");
            X1 = X1(idx1,:);
            Z1 = obj.mdl1.D(idx1,:);

            obj.idx = idx1;

            [F2,idx2] = sort(F2,"ascend");
            X2 = X2(idx2,:);
            %Z2 = obj.mdl2.D(idx,:);

            B1hat = pinv(Z1)*X1;
            X1n = Z1*B1hat;
            E1 = X1 - X1n;

            [~,~,V1] = svds(X1n,rank(X1n));
            [~,~,V2] = svds(X2n,rank(X2n));

            % Calculate, store off-diagonal prediction.
            obj.X1X2e = (X1n + E1) * V1 * obj.R * V2';
            obj.X1X2n = X1n * V1 * obj.R * V2';
            
            % Store meta-data
            obj.F1_sorted = F1;
            obj.F2_sorted = F2;
            obj.X1_sorted = X1;
            obj.X2_sorted = X2;

        end
 
    end
end
