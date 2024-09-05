# InfoFeeder
This is a project for the course Software Engineering for Data Science, the goal is to build an auto info feeder for researchers or the general public depending on the kinds of information asked for. Right now the main feature is paper feeder for researchers, whose architecture can be generalized to broader application areas.

# Why do you need this
How do you keep up with the academia? Track researchers on ResearchGate or track certain journals and browse the entire catalog to see if there are anything interesting? Either way, you still risk missing some important papers and need to spend extra time scanning the entire paper looking for results that actually matters to your research. InfoFeeder helps you keep an eye on the entire academia in fields you are interested in, track all the conferences and journals you might want to check, filter out the redundency, and summarize things you need to know with a daily/weekly/monthly feed on your preference. 

# How does it work
In general, it comprises three major steps: Scrap, Aggregate, Feed. 

## Scrap
Using arxiv or google scholar api to keep up with the frontier, scrap papers freshly published by conferences and journals, download and catagorize. 

## Aggregate
Use Chatgpt/Claude/.. text generation api, personalize a feeder(possibly for different users), select papers they might be interested in, generate a summary with important details.

## Feed
Send the feed to the user, get feedback and update preference

# Checkbox
- Find out how to use the api, what are the configurations
- Find out how good these chatbots are on summarizing research papers, and test prompts
- Find programming languages/ways to deploy the app, from backend processing and frontend feeding
- Design the logic/classes, sketch the skeleton
- Code realize the logic and design, test for errors
- Learn about frontend and typical ways to connect these two things
- Try to build the simplest frontend and make things work in the simplest way
- Add-on features, add as I go...... (how to auto-email the feeds, other useful features to personalize, etc.)

# Problems to Check
- Are there anything similar to this on the market?
- Do people really need this? (At least I do, and new researchers do because we don't know anybody or the name of any journals)
- How crap are the chatbots, any substitutions?



