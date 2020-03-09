from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sa = SentimentIntensityAnalyzer()

sa.lexicon

sa.polarity_scores("I was sent to a Costco to see if people are stocking up (even though health officials say it’s not necessary) in case COVID-19 gets more serious here. This guy came out of the store with 16 boxes of condoms and a big jar of coconut oil. We all have priorities")
sa.polarity_scores("if I was starting a new company I’d seriously consider using Rails not because I agree with most of their technical decisions (I actually disagree with a lot of them!) – but because having a Polished, Proven Universe That Works is _super_ valuable")
sa.polarity_scores("In Delhi's Connaught Place, men shouting Desh Ke Gaddaron Ko, Goli Maaron Salo Ko. marching through our capital as @DelhiPolice stands and watches. Are the deaths of our citizens already not quite enough? What is this nonsense #DelhiRiots")
sa.polarity_scores("told a group of teens that when I was their age we had to pay 10 cents per text message and now they think I’m a liar")
