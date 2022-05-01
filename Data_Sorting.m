function [Session_Data, fs] = Data_Sorting(name)
    load(name)
    session_len = size(y,1);
    cur_trial = -20;
    Session_Data = {};


    i = 1;
    is_not_over = true;
    while is_not_over

        cur_point = trig(i);
        cur_trial = cur_point;
        start_point = i;

        while trig(i) == cur_trial
                i = i+1;
            if i > session_len
                is_not_over = false;
                break
            end
        end

        Session_Data = [Session_Data;{y(start_point:i-1,:),cur_trial}];

    end
end