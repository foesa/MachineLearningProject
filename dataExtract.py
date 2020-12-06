from datetime import datetime
from searchtweets import collect_results, load_credentials, gen_rule_payload
import numpy


def auth():
    premium_args = load_credentials(filename="credentials.yaml", yaml_key='search_tweets_api_dev', env_overwrite=False)
    rule = gen_rule_payload("Drake", results_per_call=100)
    print(rule)
    tweets = collect_results(rule, max_results=100, result_stream_args=premium_args)
    [print(tweet.all_text) for tweet in tweets]


def main():
    # api = auth()
    print(time())


def time():
    startTime = datetime(2020, 6, 7).timestamp()
    endTime = datetime(2020, 11, 6).timestamp()
    randomList = numpy.random.randint(int(startTime), int(endTime), 20)
    randomDates = []
    for i in randomList:
        date = datetime.fromtimestamp(i)
        randomDates.append(date.isoformat())
    return randomDates


if __name__ == '__main__':
    main()
