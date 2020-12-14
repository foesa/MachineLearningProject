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
    
def mongo_downloader():
    """
    Purpose:
        o Downloads files from Pymongo
        o Saves them in .txt file format
    """
    CONNECTION_STRING = "mongodb+srv://foesa:Random12@cluster0.sdvcb.mongodb.net/TweetDB?retryWrites=true&w=majority"
    client = pymongo.MongoClient(CONNECTION_STRING)
    db = client.get_database('TweetDB')
    records = db.TweetsData
    tweets = records.find({})
    #length = tweets.count() 
    total_tweets=[]
    for i in tweets:
        #print(i)
        total_tweets.append(i)
    
    maxn=len(total_tweets)
    folder="project_data.txt" #Set to where you want to save file
    f=open(folder,'w',encoding='utf8')
    for i in range(0,maxn,1):
        #print(i,total_tweets[i])
        x=str(total_tweets[i])+'\n'
        f.write(x)
    f.close()

def get_dataset():
    client = pymongo.MongoClient(CONNECTION_STRING)
    db = client.get_database('TweetDB')
    records = db.TweetsData
    hash_text = {}
    skip_records = 3000  # Change this to the amount of records to skip
    retrieve_records = 3000  # Change this to the amount of records to retrieve
    tweets = records.find().skip(skip_records).limit(retrieve_records)
    sent_val = ""
    count = 0
    for i in tweets:
        print(dumps(i))
        print(i['_id'])

        if i['text'] in hash_text:
            records.update_one({"_id": i['_id']}, {"$set": {"sentiment": hash_text[i['text']]}})
            print(hash_text[i['text']])

        if 'sentiment' not in i:
            count = count +1
            print(count)
            sentiment = input('Positive(P), Negative(N), Neutral(O) or Remove(X): ').lower()
            if sentiment == 'p':
                print('Positive')
                sent_val = 'Positive'
            elif sentiment == 'n':
                print('Negative')
                sent_val = 'Negative'
            elif sentiment == 'x':
                print('Will be removed later')
                sent_val = 'Remove'
            elif sentiment == 'o':
                print('Neutral')
                sent_val = 'Neutral'

            if sent_val is not "":
                records.update_one({"_id": i['_id']}, {"$set": {"sentiment": sent_val}})
                hash_text[i['text']] = sent_val

        else:
            hash_text[i['text']] = i['sentiment']



if __name__ == '__main__':
    main()
