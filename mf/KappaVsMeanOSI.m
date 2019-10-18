function varargout = KappaVsMeanOSI(N, maxMoments, nDeltas, nThetas, ...
                                    nKappas, mExtZero, mExtOne, ...
                                    mE1Guess, Jab, Ea, cff)
%KappaVsMeanOSI(100, 10, 10, 20, 200, 0.075, .015, .8)
 
%    kappas = linspace(0, 9.5, nKappas);
       
%    kappas = 0:0.05:8; % sets kappa step = 0.05

    kappas = 0:0.5:15; % sets kappa step = 0.05    

    kappas = [kappas, linspace(15, 20, 200)];
    %kappas = [8, 3, 0]
    corr = {};
    nKappas = length(kappas)
    %    waitBarHandel =  waitbar(0);
    parpool('local', 28)
    parfor(i = 1:nKappas, 28)
    %    for i = 1:nKappas
    %        mE1Guess = mE1Guess;            
        disp(['kappa = ', num2str(kappas(i)) ]);
        %        [avg_osi(i), mE1Guess, avg_osi_I(i), mI1Guess, tmp, quantiles] ...
        
        [avg_osi(i), ~, avg_osi_I(i), mI1Guess, tmp, quantiles] ...            
            = GenOSIdistr(N, maxMoments, nDeltas, nThetas, kappas(i), ...
                          mExtZero, mExtOne, mE1Guess, Jab, Ea, cff);
        corr{i} = tmp;
        qELower(i) = quantiles.qELower;
        qEUpper(i) = quantiles.qEUpper;
        qILower(i) = quantiles.qILower;
        qIUpper(i) = quantiles.qIUpper;        

        % if mod(i, 10)
        %     message = ['\kappa = ', num2str(kappas(i))]; %, ' osi_E = ', num2str(avg_osi(i)), ' osi_I = ', num2str(avg_osi_I(i))]
        %     waitbar(i ./ nKappas, waitBarHandel, message)
        % end
    end
    delete(gcp('nocreate'))
    osi_E = avg_osi;
    osi_I = avg_osi_I;
    keyboard;
    kappa_c = Pcritical(mExtZero, Jab, Ea, cff)       % 
    filename = ['./data/kappa_vs_mZero_', num2str(1e3*mExtZero), ...
                '_mOne_', num2str(1e6*mExtOne), '_N_', num2str(N)]
    save(filename, 'kappa_c', 'kappas', 'osi_E', 'osi_I', 'qELower', 'qEUpper', ...
         'qILower', 'qIUpper')

    delete(gcp('nocreate'));
    
    varargout{1} = kappas;
    varargout{2} = avg_osi;
    varargout{3} = avg_osi_I;
    varargout{4} = corr;
    out_quantiles.qELower = qELower;
    out_quantiles.qEUpper = qEUpper;
    out_quantiles.qILower = qILower;
    out_quantiles.qIUpper = qIUpper;
    varargout{5} = out_quantiles;



    
end

