% Title :"Combining a recursive approach via non-negative matrix factorization 
%          and Gini index sparsity to improve reliable detection of
%                               wheezing sounds"
%
% Objetive: The aim of this paper is to propose a novel wheezing sound 
%           detection approach combining a recursive orthogonal non-negative 
%           matrix factorization (ONMF) and the Gini index spectral sparsity. 
%           This recursive approach is composed of four stages. The first stage 
%           is based on the ONMF model to factorize the spectral bases as 
%           dissimilar as possible. The second stage attempts to cluster the 
%           ONMF spectral bases into two categories: wheezing and normal breath. 
%           The third stage defines the stop criterion of the recursive method 
%           that avoids the loss of wheezing spectral content. Finally, the fourth
%           stage determines the patient's condition in order to locate the 
%           temporal intervals in which wheeze sounds are active for unhealthy patients.
%
%
% Journal: Expert system with applications (ESWA)
% Authors: Juan De La Torre Cruz, Francisco Jesus Canadas Quesada, Julio 
%          Jose Carabias Orti, Pedro Vera Candeas, Nicolas Ruiz Reyes

%--------------------------------------------------------------------------
clear all;close all;clc;
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%% Parameter initialization
sr = 2048;          % Sampling rate
N = 256;            % Hamming window sample length   
S = 0.25;           % Overlap
K = 80;             % Number of componets
Mi = 120;           % Number of ONMF iterations   
nfft = 2*N;         % Points of the discrete fourier transform (DFT)
rho = 0.1;          % Threshold Zs
Zh = 0.5;           % Threshold Zh (Healthy/Unhealthy)
Zd = 0.05;          % Threshold Zd (Wheezing detection)
stop = 0;           % Stop criterion initialization
Ri = 0;             % Recursive iteration initialization
%--------------------------------------------------------------------------

%-------------------------------------------------------------------------%
%                   Stage I: Obtaining ONMF bases                         %
%-------------------------------------------------------------------------%
% Select input signal x(t) (audio file .wav)
[filename,pathname] = uigetfile('*.wav','Select the input signal x(t)');
[x,fs]=audioread([pathname filesep filename]); disp([' Name: ' filename]);
% Stereo - Single
if size(x,2)==2
    x=(x(:,1)+x(:,2))/2;
end
x=x'; 
% Resample (sr=2048)
 if fs~=sr
    x=resample(x,fsre,fs); fs=sr;
 end
Lx=length(x);         % Samples
%----------------------------- Time-Frequency Representation
Hop_samples=round(S*N);                          
NFrames=floor((Lx-N+(N*S))/Hop_samples);         
noverlap=N-Hop_samples;                          
window=hamming(N);                               
[Xcomplex,fo,to]=sg(x,nfft,fs,window,noverlap);   
X=abs(Xcomplex);   % Magnitude spectrogram
nf=size(X,1);      % Bins
nt=size(X,2);      % Frames
% Normalized magnitude spectrogram
norm = sum(sum(X));
Xn = X/norm;
%===================================================================================================================%
%                                                 Figure
% imagesc((0:Lx-1)/fs,(0:nf-1)*(fs/2)/size(Xn,1),Xn);xlabel('Time(s)');ylabel('Frequency(Hz)');colormap(flipud(gray));
%===================================================================================================================%
%+++++++++++++++++++++ Start of the recursive procedure +++++++++++++++++++
while stop==0 
Ri = Ri + 1;          % recursive iteration
% Orthogonal non-negative matrix factorization (ONMF)
[B,A] = onmf(Xn, K);
% B-->  Basis matrix
% A-->  Activation matrix
%--------------------------------------------------------------------------

%-------------------------------------------------------------------------%
%                 Stage II: Clustering of the ONMF bases                  %
%-------------------------------------------------------------------------%
% Compute the sparse descriptor Gini index (beta) for each basis kth
for i=1:K
    beta(i) = ((nf+1)/nf)-((2*sum((nf:-1:1).*...
              sort(B(:,i)')))/(nf*sum(sort(B(:,i)'))));
end
% A thresholding (Zm) process is used to classify the bases into two groups
% Bw and Br.
Zm = prctile(beta,50);                % Threshold
posw = find(beta >= Zm);              
posr = find(beta < Zm);
betaW = beta(posw);                   
betaR = beta(posr);
Bw = B(:,posw);                       % Bw
Aw = A(posw,:);                       % Aw
Br = B(:,posr);                       % Br
Ar = A(posr,:);                       % Ar
% Reconstruct the estimated wheezing spectrogram Xw
Xw(:,:,Ri) = Bw*Aw;

%-------------------------------------------------------------------------%
%              Stage III: Stop criterion of the recursive method          %
%-------------------------------------------------------------------------%
% Compute the spectral difference gamma between the least periodic basis 
% betaW clustered into the wheezing bases Bw and the most periodic normal 
% breath basis betaR clustered into the normal breath bases Br
gamma = min(betaW)-max(betaR);
if Ri==1
% Compute the threshold Zs: defined to determine the optimal iteration
Zs = rho*gamma;
end
if gamma > Zs 
    stop = 0;
    Xn = Xw(:,:,Ri);
else
    stop = 1;
    Oi = Ri-1;                                      % Optimal iteration
end
end
%++++++++++++++++++++++ End of the recursive procedure ++++++++++++++++++++

% Select the optimal estimated wheezing spectrogram
Xwo = Xw(:,:,Oi);

%-------------------------------------------------------------------------%
%              Stage IV: Determine the patient's condition                %
%-------------------------------------------------------------------------%
% Compute the spectral energy distribution (epsilon)
E = sum(Xwo,2);
En = E./max(sum(Xwo,2));
% Compute the sparse descriptor Gini index (beta_e) for the spectral 
% energy distribution En
beta_e = ((nf+1)/nt)-((2*sum(((nf:-1:1).*sort(En'))))/(nf*sum(sort(En)))); 
% Healthy/Unhealthy patient
if beta_e < Zh 
     delta_t = zeros(1,nt);                    %Healthy 
     disp(' Healthy patient');
else
   % Calculate prominent estimated wheezing spectrogram Xwp
     Xwp = Xwo.^2;
     Xmax = max(Xwp);
     Xmaxn = Xmax./max(max(Xmax));
     Xi= smooth(Xmax, 3)./(max(max(smooth(Xmax,3))));
     delta_t = Xi >= Zd;                  %UnHealthy patient
     disp(' Unhealthy patient');
%=========================================================================================================================================%
%                                    Figure
% subplot(311);imagesc((0:Lx-1)/fs,(0:nf-1)*(fs/2)/size(Xn,1),X/norm);xlabel('Time(s)');ylabel('Frequency(Hz)');colormap(flipud(gray));
% subplot(312);imagesc((0:Lx-1)/fs,(0:nf-1)*(fs/2)/size(Xn,1),Xwo);xlabel('Time(s)');ylabel('Frequency(Hz)');colormap(flipud(gray));
% subplot(313);plot((0:Lx/nt:Lx-1)/fs,Xi,'k');hold on; plot((0:Lx/nt:Lx-1)/fs,delta_t,'r');hold on;plot((0:Lx/nt:Lx-1)/fs,(Xi>0)*Zd,'--k');
% legend({'$\xi(t)$','$\delta(t)$','$\zeta_{d}$'},'Interpreter','latex');
%=========================================================================================================================================%
end

