# Simple feature table (77,399 videos with a day-7 view count)

Explains % = share of the variation in day-7 views the feature accounts for on its own. Best/worst = median views in the best group divided by the worst (groups of 200+ videos).

Rule: keep a feature if it explains at least 1%, or its best group gets at least double the views of its worst. Drop it if one value covers 95% or more of videos.

| feature | explains | best/worst | most common value | where it comes from | verdict | reason |
|---|---|---|---|---|---|---|
| channel average views per video | 38.6% | 5981.7× | 42% | looked up from the channel | keep | views change clearly across its values |
| subscribers | 23.2% | 39.1× | 36% | looked up from the channel | keep | views change clearly across its values |
| YouTube topic | 10.3% | 28.7× | 25% | not available before publishing | useful, but unknown before publishing | YouTube only assigns it after a video is uploaded |
| category | 9.7% | 89.4× | 31% | on the form | keep | views change clearly across its values |
| channel video count | 4.6% | 5.5× | 40% | looked up from the channel | keep | views change clearly across its values |
| title script | 4.0% | 3.4× | 72% | worked out from the title | keep | views change clearly across its values |
| channel uploads per day | 3.8% | 4.9× | 31% | looked up from the channel | keep | views change clearly across its values |
| audio language | 3.3% | 2.9× | 53% | on the form | keep | views change clearly across its values |
| tag count | 3.0% | 2.2× | 44% | not on the form | useful, needs a new form field | views change with it, but the form doesn't ask for it |
| channel age | 2.3% | 1.8× | 42% | looked up from the channel | keep | views change clearly across its values |
| publish hour (SLT) | 1.7% | 6.2× | 7% | on the form (optional) | keep | views change clearly across its values |
| duration | 1.6% | 4.1× | 39% | on the form | keep | views change clearly across its values |
| capital letters in title | 1.2% | 3.0× | 47% | worked out from the title | keep | views change clearly across its values |
| title word count | 0.8% | 1.6× | 35% | worked out from the title | drop | views barely change across its values |
| made for kids | 0.7% | 41.7× | 99% | not on the form | drop | almost every video has the same value (99%) |
| description length | 0.7% | 1.8× | 41% | not on the form | drop | views barely change across its values |
| title length (characters) | 0.6% | 1.2× | 26% | worked out from the title | drop | views barely change across its values |
| exclamation mark in title | 0.4% | 1.6× | 93% | worked out from the title | drop | views barely change across its values |
| definition | 0.3% | 2.1× | 96% | not on the form | drop | almost every video has the same value (96%) |
| Short or long-form | 0.1% | 1.1× | 60% | should be asked on the form | keep | hidden overall because channel size is mixed in; within each size band Shorts get more views, 6 to 13 times more under 10K subscribers (SF7) |
| publish weekday | 0.1% | 1.1× | 16% | on the form (optional) | drop | views barely change across its values |
| question mark in title | 0.0% | 1.2× | 94% | worked out from the title | drop | views barely change across its values |
| number in title | 0.0% | 1.1× | 69% | worked out from the title | drop | views barely change across its values |
| captions | 0.0% |  | 100% | not on the form | drop | almost every video has the same value (100%) |
| weekend | 0.0% | 1.1× | 75% | on the form (optional) | drop | views barely change across its values |

Note: these are raw comparisons. Large channels dominate some of them, category most of all: News and Politics looks weak here only because a few channels post very many clips each. Short or long-form is the clearest case of an effect hidden this way, so its verdict is set from the channel-size split in SF7.
