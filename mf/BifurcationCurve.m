function out = BifurcationCurve(mExtZero, mExtOne, Jab, Ea, cff)
    format long
    %    cff = 1
    nPoints = 1000;
    kappa_c = Pcritical(mExtZero, Jab, Ea, cff)
    kappa = linspace(kappa_c + 5, kappa_c-1, nPoints);
    kappa = [kappa, linspace(kappa_c-1, 0, 100)];
    solutions = nan(1, length(kappa));
    disp(['kappa = ', num2str(kappa(1))])
    tmp = Solver(kappa(1), mExtZero, mExtOne, 0.08, Jab, Ea, cff);
    solutions(1) = tmp(2);
    disp(['mE^1 = ', num2str(solutions(1))])
    %    waitBarHandel =  waitbar(0);
    solutions_I = nan(1, nPoints);
    for k = 2:length(kappa)
        %        disp('---------------------------')
        %        disp(['kappa = ', num2str(kappa(k))])
        tmp = Solver(kappa(k), mExtZero, mExtOne, ...
                     solutions(k-1), Jab, Ea, cff);
        solutions(k) = tmp(2);
        solutions_I(k) = tmp(4);
        % if mod(k, 10)
        %     waitbar(k / nPoints, waitBarHandel, ['\kappa = ', num2str(kappa(k)), ...
        %                         ' m_E^{(1)} = ', num2str(solutions(k))]);
        % end
        
        %        disp(['mE1 = ', num2str(solutions(k))])
        %disp('---------------------------')
    end
    
    %    Ea = [2; 1];
    %%%%% PLOT %%%%%
    mA0 = -1 * inv(Jab) * Ea .* mExtZero;

    % figure(2)
    % hold on
    % h = plot(kappa, solutions(:) ./ mA0(1) , 'k*-')
    % h = plot(kappa, solutions_I(:) ./ mA0(2) , 'go-')
    % %    kappa_c = Pcritical(mExtZero)
    % ylim([0, 1]);
    % set(gca(), 'YTick', [0, 0.5, 1]);
    % xlim([0, 18])
    % set(gca(), 'XTick', [0, 3.5, 8]);
    % line([kappa_c, kappa_c], ylim)
    
    % hgsave(h, './figs/bc_test.png')


    mu_E = solutions ./ mA0(1);
    
    filename = ['./data/bifurcation_curve_mZero_', num2str(1e3 * mExtZero), ...
                '_mOne_', num2str(1e6 * mExtOne)]
    save(filename, 'kappa', 'mu_E', 'kappa_c')
    %    keyboard
end