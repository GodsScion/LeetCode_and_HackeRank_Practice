from random import random

def bet(bet_perc: float, money: float, probability: float) -> float:
    orig_money = money
    if probability > 100 or probability < 0:
        raise Exception("Probability must be between 0 and 100")
    if bet_perc > 100 or bet_perc < 0:
        raise Exception("Bet percentage must be between 0 and 100")
    if money < 0:
        raise Exception("Money cannot be negative")
    if random()*100 <= probability:
        money += money * bet_perc/100
    else:
        money -= money * bet_perc/100
    money = round(money, 3)
    if orig_money == money:
        raise Exception("All money's lost")
    return money 


def year(yearRun: int = 0, money: float = 1000, betPerc: float = 10, probability: float = 50, days: int = 1000):
    orig_money = money
    try:
        count = 0
        while count <= days:
            money = bet(betPerc, money, probability)
            count += 1
            # print(money, count)
        return money-orig_money
    except Exception as e:
        print(e)
        return money-orig_money
    finally:
        print(f"Result for year: {yearRun}, Profit / Loss: {money-orig_money}, Days: {count}")


def main():
    # stats
    wins = 0
    average = 0.0
    years = 12

    # year
    days = 52*5-10
    money = 1000
    betPerc = 10
    probability = 51

    print("#"*69)
    print(f'Total Streaks:      {years}\nDays per year:    {days}\nCapital:            {money}\nBet percentage:     {betPerc}\nProbability:        {probability}')
    print("-"*69)

    count = 1
    while count <= years:
        result = year(count, money, betPerc, probability, days)
        average += result
        if result > 0:
            wins += 1
        count += 1

    average = average/years
    print("-"*69)
    print(f"Total Streaks:  {years}\nWins:           {wins}\nLosses:         {years-wins}\nAvg profits:    {average}")
    print("#"*69)
main()