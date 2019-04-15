% getFeatures- Martin O'Reilly August 2018
% modified - Severin Gsponer March 2019

function [F,Fz] = getFeatures( signals )
% 1 Mean
% 2 RMS
% 3 Standard Deviation
% 4 Kurtosis
% 5 Median
% 6 Skewness
% 7 Range
% 8 Variance
% 9 Maximum
% 10 Minimum
% 11 Signal Energy
% 12 25th percentile
% 13 75th Percentile
% 14 Level crossing rate
% 15 maximum index
% 16 minimum index
% 17 number of peaks
% 18 length of segment

F=[];


for s = 1:length(signals(1,:))
    v = signals(:,s);
    y = v(1);
    v = v(2:find(v,1,'last'));
    sf=[];
    
    mn = mean(v);
    RMS = rms(v);
    sd = std(v);
    k = kurtosis(v);
    mdn = median(v);
    sk = skewness(v);
    [mx,mx_i]=max(v);
    va = var(v);
    [mnm,mnm_i]=min(v);
    rg = range(v);
    nrg=(norm(v)^2)/length(v); %signal energy
    tfp = prctile(v,25);
    sfp = prctile(v,75);
    num_peaks=length(peakfinder(v,round(1024*0.051),0.66*max(v)));
    len=length(v);   
    lc=lcr(v,mean(v));
      
    sf = [y mn RMS sd k mdn sk rg va mx(1) mnm(1) nrg tfp sfp lc  mx_i(1) mnm_i(1) num_peaks len];
    
    %sf = [mn RMS sd k mdn mod sk mnm rg va mx];
    F=[F; sf];
    
end
y = F(:,1);
s = F(:,2:end);
s = zscore(s);
Fz = [y s];


