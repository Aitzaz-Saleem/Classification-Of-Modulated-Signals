t = cputime;
y = [];
   for j = 2:101
      [~,file_path] = (xlsread('D:\Modulation_Classification\Book1.xlsx', 11 , sprintf('%s%d','A',j)));
      [~,file_name] = (xlsread('D:\Modulation_Classification\Book1.xlsx', 11 , sprintf('%s%d','B',j)));
      file_path = char(file_path);
      file_name = char(file_name);
      file = file_path + "/" + file_name;
      % Read the modulated signal from .wav file extension
      [sig, fs] = audioread(file);
%      Define the signal length
%      signal_length = length(sig);
%      Extract time-domain features
%      mean_sig = mean(sig);
%      var_sig = var(sig);
%      std_sig = std(sig);
      a = [mean(sig(:,1)) var(sig(:,1)) rms(sig(:,1)) std(sig(:,1)) kurtosis(sig(:,1)), skewness(sig(:,1)) 11];
      y = [y ; a];
   end
   xlswrite('D:\Modulation_Classification\Features.xlsx',y,1,sprintf('%s%d','A',1002))
   disp('Done');
   disp(cputime - t);
% % Define the frequency-domain features
% freq_domain = fft(sample_data);
% freq_domain = freq_domain(1:signal_length/2+1);
% freq_domain = abs(freq_domain);
% freq_domain(2:end-1) = 2*freq_domain(2:end-1);
% freq_domain_features = [mean(freq_domain), std(freq_domain), var(freq_domain)];


