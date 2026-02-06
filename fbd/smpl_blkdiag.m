function [block1, block2,off12,off21] = smpl_blkdiag(X, p, mode)
%% smpl_blkdiag (Block Diagonal Sampling)
% Subsets a block diagonal sampling from a matrix.
% [block1, block2] = smpl_blkdiag(X, p, mode) extracts two
% rectangular submatrices from the input matrix X according to the
% specified sampling percentage p. The parameter 'mode' determines how
% the matrix is partitioned:
%
%       'rows'  - Split by rows only: the first block takes the first
%                 round(m*p) rows (all columns), and the second block uses
%                 the remaining rows.
%
%       'cols'  - Split by columns only: the first block takes the first
%                 round(n*p) columns (all rows), and the second block uses
%                 the remaining columns.
%
%       'both'  - Split both rows and columns: the first block is the top
%                 left submatrix with round(m*p) rows and round(n*p) columns,
%                 and the second block is the bottom right submatrix using
%                 the remaining rows and columns.
%
% Inputs:
% X    - The input m x n matrix.
% p    - Sampling percentage (0 < p < 1). E.g., 0.3 means 30%.
% mode - A string with options: 'rows', 'cols', or 'both'.
%
% Outputs:
% block1 - The submatrix corresponding to the top left block.
% block2 - The submatrix corresponding to the bottom right block.
%
% coded by: Michael Sorochan Armstrong (mdarmstr@ugr.es)
%           Jose Camacho Paez (josecamacho@ugr.es)
%
% last modification: /Oct/2025
%
% Copyright (C) 2025  University of Granada, Granada
% Copyright (C) 2025  Jose Camacho Paez, Michael Sorochan Armstrong
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

if nargin < 3
    mode = 'both';
end

[m, n] = size(X);

switch lower(mode)

    case 'rows'
        r1 = round(m * p);

        block1 = X(1:r1, :);
        block2 = X(r1+1:end, :);

        % No meaningful off-diagonal blocks
        off12 = [];
        off21 = [];

    case 'cols'
        c1 = round(n * p);

        block1 = X(:, 1:c1);
        block2 = X(:, c1+1:end);

        % No meaningful off-diagonal blocks
        off12 = [];
        off21 = [];

    case 'both'
        r1 = round(m * p);
        c1 = round(n * p);

        % Main diagonal blocks
        block1 = X(1:r1, 1:c1);
        block2 = X(r1+1:end, c1+1:end);

        % Off-diagonal blocks
        off12  = X(1:r1, c1+1:end);     % Top-right
        off21  = X(r1+1:end, 1:c1);     % Bottom-left

    otherwise
        error('Unknown mode. Please use ''rows'', ''cols'', or ''both''.');
end
end


