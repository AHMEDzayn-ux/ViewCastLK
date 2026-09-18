# Simple feature table (70,466 videos with a day-7 view count)

Explains % = share of the variation in day-7 views the feature accounts for on its own. Best/worst = median views in the best group divided by the worst (groups of 200+ videos).

Rule: keep a feature if it explains at least 1%, or its best group gets at least double the views of its worst. Drop it if one value covers 95% or more of videos.

| feature | explains | best/worst | most common value | where it comes from | verdict | reason |
|---|---|---|---|---|---|---|
| channel average views per video | 38.3% | 5754.0× | 43% | looked up from the channel | keep | views change clearly across its values |
| subscribers | 22.5% | 37.0× | 36% | looked up from the channel | keep | views change clearly across its values |
| YouTube topic | 10.1% | 29.4× | 25% | not available before publishing | useful, but unknown before publishing | YouTube only assigns it after a video is uploaded |
| category | 9.4% | 92.0× | 31% | on the form | keep | views change clearly across its values |
| channel video count | 4.4% | 5.6× | 40% | looked up from the channel | keep | views change clearly across its values |
| title script | 3.7% | 3.3× | 72% | worked out from the title | keep | views change clearly across its values |
| channel uploads per day | 3.6% | 5.0× | 31% | looked up from the channel | keep | views change clearly across its values |
| audio language | 3.1% | 2.9× | 53% | on the form | keep | views change clearly across its values |
| tag count | 2.8% | 2.2× | 45% | not on the form | useful, needs a new form field | views change with it, but the form doesn't ask for it |
| channel age | 2.2% | 1.8× | 42% | looked up from the channel | keep | views change clearly across its values |
| publish hour (SLT) | 1.7% | 5.9× | 7% | on the form (optional) | keep | views change clearly across its values |
| duration | 1.4% | 4.0× | 39% | on the form | keep | views change clearly across its values |
| capital letters in title | 1.2% | 3.2× | 47% | worked out from the title | keep | views change clearly across its values |
| Short or long-form | 1.0% | 2.1× | 40% | should be asked on the form | keep | hidden overall by 1M+ channels; for channels under 10K subscribers Shorts get 6 to 9 times the views (SF7) |
| made for kids | 0.8% | 40.7× | 99% | not on the form | drop | almost every video has the same value (99%) |
| title word count | 0.7% | 1.6× | 35% | worked out from the title | drop | views barely change across its values |
| description length | 0.7% | 1.9× | 41% | not on the form | drop | views barely change across its values |
| title length (characters) | 0.5% | 1.2× | 26% | worked out from the title | drop | views barely change across its values |
| exclamation mark in title | 0.4% | 1.7× | 93% | worked out from the title | drop | views barely change across its values |
| definition | 0.4% | 2.3× | 96% | not on the form | drop | almost every video has the same value (96%) |
| publish weekday | 0.1% | 1.2× | 15% | on the form (optional) | drop | views barely change across its values |
| question mark in title | 0.0% | 1.3× | 94% | worked out from the title | drop | views barely change across its values |
| captions | 0.0% |  | 100% | not on the form | drop | almost every video has the same value (100%) |
| weekend | 0.0% | 1.1× | 73% | on the form (optional) | drop | views barely change across its values |
| number in title | 0.0% | 1.1× | 69% | worked out from the title | drop | views barely change across its values |

Note: these are raw comparisons. Large channels dominate some of them, category most of all: News and Politics looks weak here only because a few channels post very many clips each. Short or long-form is the clearest case of an effect hidden this way, so its verdict is set from the channel-size split in SF7.
