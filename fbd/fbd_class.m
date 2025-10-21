classdef fbd_class
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here

    properties
        X1
        F1
        X2
        F2
    end

    methods
        function obj = init(X1,F1,X2,F2)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            obj.X1 = X1;
            obj.X2 = X2;
            obj.F1 = F1;
            obj.F2 = F2;
        end

        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
    end
end