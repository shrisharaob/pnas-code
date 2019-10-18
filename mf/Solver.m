function out = Solver(kappa, mExtZero, mExtOne, mE1Guess, Jab, Ea, cff)

    %%%%% PARAMETERS %%%%%
%    Jab = [1, -1.5;
%         1, -1];
%        Ea = [2; 1];
    %%%%% MF SOLUTIONS %%%%%
    mA0 = -1.0 .* inv(Jab) * Ea .* mExtZero;
    QfuncInv = @(z) (sqrt(2) .* erfcinv(2 .* z));
    alphaA = Alpha(Jab, Ea, mExtZero, cff); 
    uA0 = -1 * sqrt(alphaA) .* QfuncInv(mA0);
    % Make a starting guess at the solution
    %    x0 = [uA0(1), mExtOne, uA0(2), mExtOne];
    %    mE1Guess = 0.08;
    x0 = [uA0(1), mE1Guess, uA0(2), mExtOne];
    % Set option to display information after each iteration
    %    options=optimset('Display','iter', 'TolX', 10e-12, 'TolFun', ...
    %                10e-12);
    options=optimset('Display','iter', 'TolX', 10e-12, 'TolFun', ...
                     10e-12);


    %    algo1 = 'levenberg-marquardt';
    algo2 = 'trust-region';
    %algo3 = 'trust-region-dogleg';
    options = optimoptions(@fsolve, 'Display','off', 'TolX', 10e-12, 'TolFun', ...
                     10e-12, 'FiniteDifferenceType', 'central', ...
                           'MaxFunctionEvaluations', 20000, ...
                           'MaxIterations', 20000, ...
                           'Algorithm', algo2);

    
    myfunc = @(y) Moments(y, Jab, Ea, kappa, mExtZero, mExtOne, uA0, alphaA, mA0);
    % Solve the system
    [x,fval,exitflag] = fsolve(myfunc, x0, options);
    x(2) = abs(x(2)); %mE1
    out = x;
    %    disp('######################################################')
    %disp(['kappa = ', num2str(kappa), ' mI_1 = ', num2str(x(4))]);
    %disp('######################################################')    
end
