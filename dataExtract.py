from bson.json_util import dumps
from datetime import datetime, timedelta
from searchtweets import collect_results, load_credentials, gen_rule_payload
import numpy
import pymongo

CONNECTION_STRING = "mongodb+srv://foesa:Random12@cluster0.sdvcb.mongodb.net/TweetDB?retryWrites=true&w=majority"


# TODO: Change endpoint to archive, figure out how to stop @'s being included
def auth(dates):
    premium_args = load_credentials(filename="credentials.yaml", yaml_key='search_tweets_api_dev', env_overwrite=False)
    # Change the below string to the candidate you're looking for info on. Don't remove the lang:en otherwise you'll
    # get results in any language
    queryString = 'Donald Trump lang:en'
    rule = gen_rule_payload(queryString, results_per_call=100, from_date=dates[0], to_date=dates[1])
    print(rule)
    tweets = collect_results(rule, max_results=100, result_stream_args=premium_args)
    [print(tweet.all_text) for tweet in tweets]
    return tweets, queryString


def main():
    # dateList = time()
    # iterate = 0
    # for i in dateList:
    #     tweets, query = auth(i)
    #     mongo_uploader(tweets, query)
    #     print('Getting next tweets for new time, iteration ', iterate, '\n\n\n\n')
    #     iterate = iterate + 1
    get_dataset()


def time():
    startTime = datetime(2020, 6, 7).timestamp()
    endTime = datetime(2020, 11, 6).timestamp()
    # change the 20 at the end to the number of requests you wish to make. (Max 100 requests altogether)
    randomList = numpy.random.randint(int(startTime), int(endTime), 5)
    randomDates = []
    for i in randomList:
        date = datetime.fromtimestamp(i)
        endDate = date + timedelta(minutes=1)
        randomDates.append((date.isoformat(timespec='minutes'), endDate.isoformat(timespec='minutes')))

    return randomDates


def mongo_uploader(tweets, query, dates=None):
    client = pymongo.MongoClient(CONNECTION_STRING)
    db = client.get_database('TweetDB')
    records = db.TweetsData
    for tweet in tweets:
        sampleTweet = {
            "text": tweet.all_text,
            "date": tweet.created_at_datetime,
            "Candidate": query
        }
        records.insert_one(sampleTweet)
    print(records.count_documents({}))


    # TODO Need to figure out how to prevent duplicate dates. Not a huge problem but still
    # if dates is not None:
    #     for date in dates:
    #         records.count_documents({})


def get_dataset():
    client = pymongo.MongoClient(CONNECTION_STRING)
    db = client.get_database('TweetDB')
    records = db.TweetsData
    skip_records = 3000  # Change this to the amount of records to skip
    retrieve_records = 3000  # Change this to the amount of records to retrieve
    tweets = records.find().skip(skip_records).limit(retrieve_records)
    for i in tweets:
        if 'sentiment' not in i:
            sentiment = input('Positive(P), Negative(N), Neutral(O) or Remove(X): ').lower()
            if sentiment == 'p':
                print('Positive')
                records.update_one({"_id": i['_id']}, {"$set": {"sentiment": 'Positive'}})
            elif sentiment == 'n':
                print('Negative')
                records.update_one({"_id": i['_id']}, {"$set": {"sentiment": 'Negative'}})
            elif sentiment == 'x':
                print('Will be removed later')
                records.update_one({"_id": i['_id']}, {"$set": {"sentiment": 'Remove'}})
            elif sentiment == 'o':
                print('Neutral')
                records.update_one({"_id": i['_id']}, {"$set": {"sentiment": 'Neutral'}})


if __name__ == '__main__':
    main()
