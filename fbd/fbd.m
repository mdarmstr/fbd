classdef fbd < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here

    properties
        X1            = []
        F1            = []
        X2            = []
        F2            = []
        params struct = struct()
        mdl1          = []
        mdl2          = []
        congp         = []
        tbl1          = []
        tbl2          = []
        T1oe          = []
        T1r           = []
        T2oe          = []
        R             = []
        X1X2e         = []
        X1X2n         = []
    end

    methods
        function obj = fbd(X1, F1, X2, F2, params)
            % --- Required arguments ---
            if nargin < 4
                error('FBD requires X1, F1, X2, and F2.');
            end

            % --- Add paths (helpers + MEDA) ---
            addpath(genpath('.'));
            addpath(genpath('../MEDA'));
            if ~license('test', 'Statistics_Toolbox')
                error('Statistics and Machine Learning Toolbox not available.');
            end

            % --- Store data ---
            obj.X1 = X1;
            obj.F1 = F1;
            obj.X2 = X2;
            obj.F2 = F2;

            % --- Handle params ---
            if nargin < 5 || isempty(params)
                % Use default parameter structure
                obj.params = struct( ...
                    'Preprocessing', 1, ...
                    'NumPerm',       1000, ...
                    'Factors',       {{1,2,[1,2]}}...
                );
                disp('Initialized with default parameters.');
            else
                % Use provided parameter struct
                if ~isstruct(params)
                    error('Parameter input must be a struct.');
                end
                obj.params = params;
                disp('Initialized with user-specified parameters.');
            end
        end

        function test_factors(obj)
            
            %test for interactions
            %note that subsetting linear factors is not possible at the
            %modelling stage - trim F1, F2 to proceed.
            intr = cellfun(@(x) numel(x) > 1,obj.params.Factors);

            if any(intr)
                disp('Testing linear factors, interactions')
                intr_list = obj.params.Factors(intr);
                [obj.tbl1,obj.mdl1] = parglm(obj.X1,obj.F1,...
                'Model', intr_list,...
                'Preprocessing',obj.params.Preprocessing,...
                'Permutations', obj.params.NumPerm ...
                );

            [obj.tbl2,obj.mdl2] = parglm(obj.X2, obj.F2,...
                'Model', intr_list,...
                'Preprocessing', obj.params.Preprocessing,...
                'Permutations', obj.params.NumPerm ...
                );
            else
                disp('Testing linear factors only')
                [obj.tbl1,obj.mdl1] = parglm(obj.X1,obj.F1,...
                'Model', 'linear',...
                'Preprocessing',obj.params.Preprocessing,...
                'Permutations', obj.params.NumPerm ...
                );                
            [obj.tbl2,obj.mdl2] = parglm(obj.X2, obj.F2,...
                'Model', 'linear',...
                'Preprocessing', obj.params.Preprocessing,...
                'Permutations', obj.params.NumPerm ...
                );
            end

            disp(obj.tbl1)
            disp(obj.tbl2)
        end

        function test_congruence(obj)

            [obj.congp,obj.T1oe,obj.T1r,obj.T2oe,obj.R] = nnspt(obj.mdl1,obj.mdl2,obj.F1,obj.F2,...
                obj.params.Factors,obj.params.NumPerm);
            
        end

        function test_power(obj)
            disp(obj)
            disp('placeholder')
        end
        
        function pred_X1X2(obj)
            %(X_1n + E_1)P_1O_1>2P_2^T
            %Check for interactions - sum model accordingly.
            

            if any(cellfun(@(x) numel(x) > 1,obj.params.Factors)) %check for interactions
                is_intr = cellfun(@(x) numel(x) > 1, obj.params.Factors);
                is_term = cellfun(@(x) isscalar(x), obj.params.Factors);
                X1r = zeros(size(obj.mdl1.factors{1}.matrix));
                X2r = zeros(size(obj.mdl2.factors{1}.matrix));

                for ii = 1:length(is_term)
                    X1r = X1r + obj.mdl1.factors{ii}.matrix;
                    X2r = X2r + obj.mdl2.factors{ii}.matrix;
                end

                for ii = 1:length(is_intr)
                    X1r = X1r + obj.mdl1.interactions{ii}.matrix;
                    X2r = X2r + obj.mdl2.interactions{ii}.matrix;
                end

            else 
                is_term = cellfun(@(x) numel(x) > 1, obj.params.Factors);
                X1r = zeros(size(obj.mdl1.factors{1}.matrix));
                X2r = zeros(size(obj.mdl2.factors{1}.matrix));

                for ii = 1:length(is_term)
                    X1r = X1r + obj.mdl1.factors{ii}.matrix;
                    X2r = X2r + obj.mdl2.factors{ii}.matrix;                    
                end

            end
            
            [~,~,V1] = svds(X1r,rank(X1r));
            [~,~,V2] = svds(X2r,rank(X2r));

            obj.X1X2e = (X1r + obj.mdl1.residuals) * V1 * obj.R * V2';
            obj.X1X2n = X1r * V1 * obj.R * V2';

        end
 
    end
end
