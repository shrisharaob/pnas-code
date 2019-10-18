function varargout = GenOSIdistr_vs_beta(N, maxMoments, nDeltas, nThetas, ...
                                 kappa, mExtZero, mExtOne, mE1Guess, ...
                                         Jab, Ea, cff, jii_beta)

    Jab(1, 2) = Jab(1, 2) / jii_beta;
    Jab(2, 2) = Jab(2, 2) / jii_beta;
    

    [tcE, mE1_sol, tcI, mI1_sol, corr, thetas, phiFF] = GenTuningCurves(N, ...
                                                      maxMoments, ...
                                                      nDeltas, nThetas, kappa, mExtZero, mExtOne, mE1Guess, Jab, Ea, cff);
    osi_of_pop = OSIOfPop(tcE);
    varargout{1} = nanmean(osi_of_pop);

    osi_of_pop_I = OSIOfPop(tcI);
    varargout{3} = nanmean(osi_of_pop_I);
    
    %%quantiles
    quantiles.qELower = quantile(osi_of_pop, 0.25);
    quantiles.qEUpper = quantile(osi_of_pop, 0.75);
    quantiles.qILower = quantile(osi_of_pop_I, 0.25);
    quantiles.qIUpper = quantile(osi_of_pop_I, 0.75);    
    
    varargout{2} = mE1_sol;
    varargout{4} = mI1_sol ;
    varargout{5} = corr;
    varargout{6} = quantiles;


    osi_E = osi_of_pop;
    osi_I = osi_of_pop_I;
    
    %    saveOut.osiE = osi_of_pop;
    %saveOut.osiI = osi_of_pop_I;
    %saveOut.quantiles = quantiles;
    %saveOut.thetas = thetas;
    %saveOut.phiFF = phiFF;

   filename = ['./data/OSI/osi_mZero_', num2str(mExtZero * 1e3), ...
               '_mOne_', num2str(1e4*mExtOne), '_kappa_', num2str(kappa), '_N', ...
          num2str(N), '_jII_beta_', num2str(jii_beta * 1e3)]
    save(filename, 'osi_E', 'osi_I')     

    %    keyboard;


    %    save(['./data/OSI/osi_mZero', num2str(mExtZero), '_mOne_', ...
    %     num2str(1e4*mExtOne), '_kappa_', num2str(kappa), '_N', ...
    %     num2str(N), '_jII_beta_', num2str(jii_beta * 1e3)], ...
    %    'osi_E', 'osi_I')     
    %    
    
    
    %    save(['./data/OSI/osi_mZero', num2str(mExtZero), '_mOne_', ...
    %     num2str(1e4*mExtOne), '_kappa_', num2str(kappa), '_N', ...
    %     num2str(N), '_jII_beta_', num2str(jii_beta * 1e3)], 'saveOut')    
    %    tc.E = tcE;
    % tc.I = tcI;                                                                     
    %save(['./data/TC/tc_kappa_', num2str(10*kappa), '_mExtZero_', ...
    %     num2str(1e3*mExtZero), '_mOne_', num2str(1e4*mExtOne), '_N', num2str(N)], 'tc')    
    

    disp(['mean osi E = ', num2str(nanmean(osi_of_pop)), ' std: ', num2str(nanstd(osi_of_pop))]);
    disp(['mean osi I = ', num2str(nanmean(osi_of_pop_I)), ' std: ', ...
          num2str(nanstd(osi_of_pop_I))]);    

    disp(['mean m0 E = ', num2str(mean(mean(tcE)))])
    disp(['mean m0 I = ', num2str(mean(mean(tcI)))])


    
    [counts,edges] = histcounts(osi_of_pop, 50);
    plot(edges(1:end-1), counts, 'k');
    hold on
    [counts,edges] = histcounts(osi_of_pop_I, 50);
    plot(edges(1:end-1), counts, 'r');
    title('osi')
    xlim([0, 1])
    %    saveas(gcf, ['./figs/osi_m1_', num2str(mExtOne), '.png'])

    disp('exiting GenOSIdistr')
end

function osi = OSI(firingRate, atTheta)
    osi = nan;
    zk = firingRate * exp(2j * atTheta');
    if(mean(firingRate) > 0.0);
        osi = abs(zk) / sum(firingRate);
    end
end

function osi_of_pop =  OSIOfPop(firingRates)
    [nNeurons, nThetas] = size(firingRates);
    atThetas = linspace(0, pi, nThetas);
    %osi_of_pop = zeros(nNeurons, 1);
    for k = 1:nNeurons
        osi_of_pop(k) = OSI(firingRates(k , :), atThetas);
    end
end


    
    
    
    