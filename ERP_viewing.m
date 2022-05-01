close all
Nyq = fs / 2;


filtfilt_order = 4;
lf = 1;
hf = 4;
[myButter.b, myButter.a] = butter(filtfilt_order, [lf hf]/Nyq,'bandpass');

trials_len = length(Session_Data);

for ch = 1:16

    erp_left = [];
    erp_right = [];

    for i =1:trials_len

        if Session_Data{i,2} ~= 0
            trial_to_plot = filtfilt(myButter.b, myButter.a, Session_Data{i,1}(:,ch));
            
            if Session_Data{i,2} == 1
                erp_left = [erp_left,trial_to_plot];
            else
                erp_right = [erp_right,trial_to_plot];
            end

        end




    end

subplot(4,4,ch)
plot((1:length(erp_left))/fs,mean(erp_left,2),'r')
hold on
plot((1:length(erp_left))/fs,mean(erp_right,2),'b')
title(['Channel ' num2str(ch)])
ylabel('uV')
xlabel('seconds')
xlim([526/fs,(512+256)/fs])

end