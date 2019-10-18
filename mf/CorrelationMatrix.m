function [corr_mat, varargout] = CorrelationMatrix(maxMoments, ...
                                                   nDeltas, kappa, ...
                                                   Jab, theta, ...
                                                   mExtZero, mExtOne, ...
                                                   mE1Guess, Ea, cff)
% the correlations are functions only of delta for the case with no map
    %%                                               
    %    Ea = [2; 1];
    qA0_of_delta = QofDelta(nDeltas, kappa, theta, mExtZero, mExtOne, ...
                            mE1Guess, Jab, Ea, cff);
    deltas = linspace(0, pi, nDeltas);
    %%%%% COMPUTE MOMENTS OF beta IN DELTA %%%%%
    moments_of_beta = zeros(2, maxMoments);
    beta_of_delta = DefBeta(Jab, Ea, mExtZero, mExtOne, qA0_of_delta, ...
                            0, deltas, cff);
    for m = 1:maxMoments
        moments_of_beta(:, m) = MomentsOfBeta(beta_of_delta, deltas, m-1);
    end
   %%%%% CONSTRUCT CORRELATION MATRIX %%%%%
    corr_mat = zeros(maxMoments - 1);
    corr_mat_I = zeros(maxMoments-1);
    for i = 1:maxMoments
        for j = 1:maxMoments
            % the correlation matrix is diagonal when there is no map
            if i == j
                corr_mat(i, j) = moments_of_beta(1, i);
                corr_mat_I(i, j) = moments_of_beta(2, i);
            end
        end
    end
    varargout{1} = corr_mat_I;
end         

%%
function qA0_of_delta = QofDelta(nDeltas, kappa, theta, mExtZero, ...
                                 mExtOne, mE1Guess, Jab, Ea, cff)
    deltas = linspace(0, pi, nDeltas);
    qA0_of_delta = nan(2, nDeltas);
    for i = 1:nDeltas
        qA0_of_delta(:, i) = SolveBeta(kappa, theta, deltas(i), ...
                                       mExtZero, mExtOne, mE1Guess, ...
                                       Jab, Ea, cff);
    end
end


function out = MomentsOfBeta(beta_of_delta, deltas, m)
    if m == 0
        out(1) = trapz(deltas, beta_of_delta(1, :)) / pi;
        out(2) = trapz(deltas, beta_of_delta(2, :)) / pi;        
    else
        out(1) = 2 * trapz(deltas, beta_of_delta(1, :) .* cos(4.0 * m * deltas)) / pi;
        out(2) = 2 * trapz(deltas, beta_of_delta(2, :) .* cos(4.0 * m * deltas)) / pi;
    end
        out = abs(out);
end

function y = KroneckerDelta(i, j)                             
    y = 0;
    if i == j
        y = 1;
    end
end
