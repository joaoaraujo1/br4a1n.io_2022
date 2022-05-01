close all
Nyq = fs / 2;


filtfilt_order = 4;
lf = 4;
hf = 90;
[myButter.b, myButter.a] = butter(filtfilt_order, [lf hf]/Nyq,'bandpass');

trials_len = length(Session_Data);

for ch = 1:16

    pxx_left = [];
    pxx_right = [];

    for i =1:trials_len

        if Session_Data{i,2} ~= 0
            trial_to_plot = filtfilt(myButter.b, myButter.a, Session_Data{i,1}(:,ch));
            [pxx,f] = pwelch(trial_to_plot,hann(2048),0,[],fs);
        end

        if Session_Data{i,2} == 1
            pxx_left = [pxx_left,mean(pxx,2)];
            
        elseif Session_Data{i,2} == -1
            pxx_right = [pxx_right,mean(pxx,2)];
        end


    end

subplot(4,4,ch)
plot(f,mean(pxx_left,2),'r')
hold on
plot(f,mean(pxx_right,2),'k')
title(['Channel ' num2str(ch)])
xlim([3,95])
ylim([0,10])
ylabel('uV')
xlabel('Hz')

end
