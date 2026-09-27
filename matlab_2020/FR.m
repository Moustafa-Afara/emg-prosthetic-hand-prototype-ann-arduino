function [ answ ] = FR( binned_signal )
% Frequency Ratio
%
% Ratio of the smallest to the largest FFT magnitude of each column
% (the 2020 comment described a frequency ratio; the code computes this magnitude ratio).

[R, C] = size(binned_signal);
answ = zeros(1,C);
L = R; % Number of samples
NFFT = 2^nextpow2(L); % Next power of 2 from length of y
for i = 1:C
    Fy = abs(fft(binned_signal(:,i),NFFT)/L);
    answ(1,i) = min(Fy)/max(Fy); 
end

end

