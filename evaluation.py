import datetime

'''
一次计算全部指标，附加的accuracy精确到天，即计算日期相同的比例
'''
def calculateAllMetrics(real_signed_time_array, pred_signed_time_array):
    if len(real_signed_time_array) != len(pred_signed_time_array):
        print("[Error!] in calculateAllMetrics: len(real_signed_time_array) != len(pred_signed_time_array)")
        return -1

    score_accumulate = 0
    onTime_count = 0
    correct_count = 0
    total_count = len(real_signed_time_array)    
    for i in range(total_count):
        real_signed_time = datetime.datetime.strptime(real_signed_time_array[i], "%Y-%m-%d %H:%M:%S")
        pred_signed_time = datetime.datetime.strptime(pred_signed_time_array[i], "%Y-%m-%d %H:%M:%S")
        time_interval = int((real_signed_time - pred_signed_time).total_seconds() / 3600)

        # rankScore
        score_accumulate += time_interval**2

        # onTimePercent
        if pred_signed_time.month < real_signed_time.month:
            onTime_count += 1
        elif pred_signed_time.month == real_signed_time.month:
            if pred_signed_time.day <= real_signed_time.day:
                onTime_count += 1

        # accuracy
        if real_signed_time.year == pred_signed_time.year and real_signed_time.day == pred_signed_time.day:
            correct_count+=1

    accuracy = float(correct_count/total_count)
    onTimePercent = float(onTime_count/total_count)
    rankScore = float((score_accumulate/total_count)**0.5)

    # print("==> [in evaluation] rankScore is {}, onTimePercent is {}, accuracy is {}".format(rankScore,onTimePercent,accuracy))
    return (rankScore,onTimePercent,accuracy)

def test():
    pass

