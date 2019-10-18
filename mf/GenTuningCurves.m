function varargout = GenTuningCurves(N, maxMoments, nDeltas, nThetas, ...
                                     kappa, mExtZero, mExtOne, ...
                                     mE1Guess, Jab, Ea, cff)
    % GenTuningCurves(100, 10, 100, 36, 0, 0.075, 0.01, 0, [1, -1.5, 1, -1], [2; 1], 1)
    disp('entering gentuning curves')
    %    Ea = [2; 1];
    %%%% Check if parameters stisfy balance conditions %
    CheckBalCond(Jab, Ea)
    %%%%% MF SOLUTIONS %%%%%
    mA0 = -1.0 .* inv(Jab) * Ea .* mExtZero;
    QfuncInv = @(z) (sqrt(2) .* erfcinv(2 .* z));
    alphaA = Alpha(Jab, Ea, mExtZero, cff);
    uA0_tmp = -1 * sqrt(alphaA) .* QfuncInv(mA0);
    out_tmp = Solver(kappa, mExtZero, mExtOne, mE1Guess, Jab, Ea, cff);
    uE0 = out_tmp(1);
    mE1 = out_tmp(2);
    uI0 = out_tmp(3);
    mI1 = out_tmp(4);
    %
    uA0 = [uE0; uI0];
    qA0 = SolveBeta(kappa, 0, 0, mExtZero, mExtOne, mE1Guess, Jab, ...
                    Ea, cff);
    %% compute correlation matrix
    tcE = nan(N, nThetas);
    tcI = nan(N, nThetas);
    [corrMat, corrMat_I] = CorrelationMatrix(maxMoments, nDeltas, ...
                                             kappa, Jab, 0, mExtZero, ...
                                             mExtOne, mE1Guess, Ea, cff);
    cholMat = chol(corrMat);
    cholMat_I = chol(corrMat_I);
    disp('q solved');
    %% 
    s = rng(4321);  % for reproducable tuning curves
    phiFF = zeros(1, N);
    %%
    xt_l = [];
    % xa = linspace(0, pi, nThetas)
    %waitBarHandel =  waitbar(0/N, 'generating tuning curves: pc
    %done');
    %parpool('local', 28)
    %parfor (i = 1:N, 28)
    for i = 1:N
        % PO FF
        phiFF(i) =  pi * rand(); 
        % if i % 10:
        %waitbar(i/N, waitBarHandel);
            %            disp(['pc done: ', num2str(100 * i / N), '%'])
            %end
        thetas = linspace(0, pi, nThetas) - phiFF(i);
        % Excitatory
        x = randn(maxMoments, 1);
        y = randn(maxMoments, 1);
        v = cholMat * x;
        w = cholMat * y;
        tmp = repmat((0:maxMoments-1), nThetas, 1);
        tmp2 = repmat(thetas', 1, maxMoments);
        cosVector = cos(2 * tmp .* tmp2);
        sinVector = sin(2 * tmp .* tmp2);
        x_of_theta = v' * cosVector' + w' * sinVector';
        
        xt_l = [xt_l; x_of_theta];
        
        % Inhibitory
        xi = randn(maxMoments, 1);
        yi = randn(maxMoments, 1);
        vi = cholMat_I * xi; 
        wi = cholMat_I * yi;
               tmp = repmat((0:maxMoments-1), nThetas, 1);
        tmp2 = repmat(thetas', 1, maxMoments);
        cosVector = cos(2 * tmp .* tmp2);
        sinVector = sin(2 * tmp .* tmp2);
        xi_of_theta = vi' * cosVector' + wi' * sinVector';
        xei = [x_of_theta; xi_of_theta];
%         
        %%%%%
        tuning_curves = Miofx(xei, Jab, Ea, kappa, mExtZero, mExtOne, ...
                              uA0, alphaA, mE1, qA0, thetas, cff);
        tcE(i, :) = tuning_curves(1, :); 
        tcI(i, :) = tuning_curves(2, :);   

        %%
%         xaxis = linspace(0, 180, nThetas); 
%         plot(xaxis, tcE(i, :), 'k')
%         osi = OSI(tcE(i, :), xaxis'*pi/180);
%         title(num2str(osi));
%         hold on
%         plot(xaxis, tcI(i, :), 'r')
%         line([phiFF(i), phiFF(i)] .* 180/pi, ylim())
%         waitforbuttonpress
%         clf
    end
    
    varargout{1} = tcE; %[tcE; tcI];
    po_of_E = POofPop(tcE);
    po_of_I = POofPop(tcI);
    po_of_FF = phiFF * 180 / pi;
    
    PLOT_PO_SCATTER = 0;
    if PLOT_PO_SCATTER
    figure(103)
    rdx = 1:2000;
    plot(po_of_FF(rdx), po_of_I(rdx), 'ro');
    xlim([0, 180])
    ylim([0, 180])
    %%
     figure(105)
     plot(po_of_FF(rdx), po_of_E(rdx), 'ko')
     xlabel('PO FF');                    
     ylabel('PO response');
     title(['m^{(1)} = ', num2str(mExtOne),  '  c_{ff}=', ...
          num2str(cff)]);
             title(['m^{(1)} = ', num2str(mExtOne),  '  c_{ff}=', ...
              num2str(cff)]);
     %        title(['c_{ff}=', num2str(cff)]);            
                 xlim([0, 180])
                 ylim([0, 180])


                 keyboard
                    %     keyboard                    
     saveas(gcf, ['./figs/po_corr_m1_',  'c_ff=', num2str(cff), '_', ...
                    num2str(mExtOne), '.png'])
     %          keyboard;    
%     
    save(['./data/poff_and_po_EandI_m1_', num2str(floor(1e4*mExtOne)), '_cff_', num2str(1e3*cff)], ...
         'po_of_FF', 'po_of_E', 'po_of_I')
    figure()
    end
    %%
    varargout{2} = mE1;
    varargout{3} = tcI; %[tcE; tcI];
    varargout{4} = mI1;
    corr.E = cholMat;
    corr.I = cholMat_I;
    varargout{5} = corr;
    %varargout{6} = thetas;
    varargout{7} = phiFF * 180/pi;
    disp('exiting gentuning curves');
    %    delete(gcp('nocreate'));
end


function osi = OSI(firingRate, atTheta)
    osi = nan;
    zk = firingRate * exp(2j * atTheta);
    if(mean(firingRate) > 0.0);
        osi = abs(zk) / sum(firingRate);
    end
end
        