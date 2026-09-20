"""Interpretation text for viewership_eda.ipynb, written against the 18 September 2026 build.

The numbers inside the notebook's charts and tables are recomputed on every
run; these notes are not. Re-read them after rebuilding on newer data.
"""

NOTES = {
    # ------------------------------------------------------------ overview
    "overview": """
**What this shows.** About 124,000 videos from roughly 2,500 Sri Lankan channels,
published from 2 July to 18 September 2026. Around 4,000 are left out (live
streams, and videos that had no duration because they were removed). Each
horizon has about 73,000 to 77,000 labelled videos, and about 44,000 videos have
all four labels.
""",
    "types": """
**What this shows.** Almost every input is complete. The exceptions are audio
language (not set on about 8% of videos) and category (under 1%). Channel
statistics look complete here, but for about 31% of videos they were recorded
only after publishing (see section 2), so the channel charts leave those out.
""",
    "per_day": """
**What this shows.** About 1,400 to 1,700 videos are published per day, steady
across the whole period. Videos from the first two weeks mostly have no day-7
label because tracking had not started for them yet. The last week has no
day-7 label yet because those videos are not 7 days old.
""",
    "quality": """
**What this shows.**

* **Channel statistics recorded after publishing** (31% of videos) are all from
  videos published before 1 August: every video in the first two weeks, 87% in
  the third and about half in the next two. None after that. For those videos we do
  not know the channel's size on publishing day, so channel charts leave them
  out rather than guess.
* **Labels are close to the exact day.** The typical label is 1.5 hours away
  from exactly 7, 14, 21 or 30 days, over 90% are within 3 hours, and none is
  more than 12 hours away.
* **Titles and descriptions are rarely edited** (about 1 to 2%), so the title
  we recorded is almost always the one the video was published with.
* **The Short flag** comes from the video's player shape for 98% of videos.
""",
    "target": """
**What this shows.** Views are extremely uneven. The typical video gets about
1,200 views in its first week, a quarter get under about 200, and the top 1%
get over 340,000. On a normal scale (left) almost every video is squashed
against zero. On a log scale (right) the shape is easy to read, which is why
every chart in this notebook uses ratios and log scales.
""",
    "growth_intro": """
**What this shows.** Most views arrive in the first week. The typical video's
day-30 views are only a few per cent above its day-7 views. The day-7 forecast
is therefore close to the final number for most videos; later horizons add
little for a typical video but more for the minority that keep growing
(section 12).
""",

    # ---------------------------------------------------------- channel
    "section_channel": """
These describe the channel, not the video. A creator cannot change them before
posting, but they are the strongest inputs by far. Each video's channel figures
are taken on its publishing day. There is no orange diamond here, because a
channel's inputs are the same for all its videos.
""",
    "subscribers": """
**What it shows.** The bigger the channel, the more views, up to about 100K
subscribers. A channel under 1K subscribers typically gets about 60 views a
week; one with 100K or more gets about 2,200. Above 100K the typical video
stops rising: 100K to 1M and 1M+ channels are about the same. **Verdict: keep.**
""",
    "subscribers horizons": """
**Over time.** The pattern is the same at day 7 and day 30. Small channels gain
slightly relative to big ones over the month, but the order never changes.
""",
    "channel average views per video": """
**What it shows.** The single strongest input: 38% of the variation on its own.
A channel whose past videos averaged over 100K views typically gets about
29,000 views on a new video; one averaging under 100 gets about 5. It works
better than subscribers because it measures how many people actually watch,
not how many once subscribed. **Verdict: keep.**
""",
    "channel video count": """
**What it shows.** Channels with more uploads get more views, up to about
10,000 videos. Beyond that, views per video fall slightly, because those are
the very high-volume channels (mostly news) that split their audience across
many clips. **Verdict: keep.**
""",
    "channel uploads per day": """
**What it shows.** Channels posting under one video every ten days get the
fewest views. Views rise with posting rate up to 2 to 10 videos a day, then
drop for channels posting more than 10 a day (again, mostly news). It overlaps
heavily with video count (section 16). **Verdict: keep.**
""",
    "channel age": """
**What it shows.** A weak effect. Channels older than 5 years get a little
more than younger ones, but the difference is under 2×. **Verdict: keep** (it
explains 2.5%, over the 1% bar), but it adds little once size is known.
""",
    "YouTube topic": """
**What it shows.** The topic YouTube assigns to the **channel** (for example
Film, Politics, Knowledge). Film and Entertainment channels typically get
about 3.5× the typical video; Knowledge channels about 0.13×. It explains 10%
on its own.

**Important:** this comes from the channel lookup the API already makes, so it
*is* available before publishing. The API does not send it today. **Verdict:
keep, and fetch it at forecast time.**
""",
    "constants_channel": """
Almost every channel lists Sri Lanka, so country carries no information.
**Verdict: drop.**
""",

    # ----------------------------------------------------------- format
    "section_format": """
These describe the video itself, and a creator decides them. From here on, the
orange diamond (same channel) matters most.
""",
    "duration": """
**What it shows.** The clearest video-level input. Videos of 20 to 60 minutes
get about 3× the typical video, and it holds within the same channel: videos
of 20 minutes or more get about 1.7× to 1.9× the channel's average. Videos of **1 to 3 minutes do worst** in both
views. Many of them are short news clips, and some are horizontal clips that
are not Shorts. **Verdict: keep (strong).**
""",
    "duration horizons": """
**Over time.** Long videos gain a little more after the first week than short
ones (60+ minutes rises from 1.0× to 1.2×), but the order stays the same.
""",
    "Short or long-form": """
**What it shows.** Overall, Shorts and long-form look almost the same (1.3K
against 1.1K typical views). Compared with their own channel, Shorts do about
1.17× better. That overall number hides a big split by channel size, shown in
the next two charts. **Verdict: keep, and add a Short / long-form field to the
form.**
""",
    "Short or long-form horizons": """
**Over time.** On the same videos, Shorts stay at about 1.2× and long-form at
about 0.8× from day 7 to day 30. A Short's advantage does not fade later.
""",
    "shorts_by_size": """
**What it shows.** Within every size band, Shorts get more views than
long-form: 13× more for channels under 1K subscribers and 6× for 1K to 10K.

The second row compares Shorts with long-form **on the same channel**, which is
the fair comparison. For channels under 100K subscribers, Shorts beat the
channel's own long-form by about 1.35× to 2×. For channels over 100K, Shorts
do slightly worse. So "post Shorts" is good advice for small and mid-size
channels, and not for big ones.
""",
    "category": """
**What it shows.** Comedy (4.7×), Entertainment (3.3×) and People & Blogs
(1.9×) get the most views; Howto & Style (0.19×) and Science & Technology
(0.41×) the fewest. This raw ranking is partly about which channels post in
each category (section 10 shows this). **Verdict: keep.**
""",
    "category horizons": """
**Over time.** Mostly stable. Education and Howto & Style gain relative to
others after the first week (people search for them later), while News &
Politics loses a little (news is watched on the day).
""",
    "constants_format": """
HD or SD, captions and made for kids are the same for 96% or more of videos,
and are not on the form. **Verdict: drop all three.**
""",

    # --------------------------------------------------------- language
    "section_language": """
Language inputs vary within a channel more than you might expect: about a
third of channels post in more than one audio language.
""",
    "audio language": """
**What it shows.** Tamil (1.4×) and Sinhala (1.15×) videos get more views than
English (1.0×). Videos with no audio language set do worst (0.22×), probably
because careful uploaders set it. Within the same channel, English videos do a
little better than the channel's other languages. **Verdict: keep.**
""",
    "default language": """
**What it shows.** Similar to audio language, and they agree on only 58% of
videos. The API sends audio language mapped to en / si / ta, so the model
should be trained on that same mapping (section 17). **Verdict: keep one;
audio language is the one the form collects.**
""",
    "title script": """
**What it shows.** Titles in Tamil script (1.7×) and Sinhala script (1.2×) do
better than Latin-script titles (0.5×). Within a channel the difference almost
disappears (1.05×): the raw gap is mostly which channels write in which script.
It passes the rule only through the biggest channels. **Verdict: keep, weak.
It is worked out from the title, which the form already collects.**
""",

    # ----------------------------------------------------------- timing
    "section_timing": """
All times are Sri Lanka time.
""",
    "publish hour": """
**What it shows.** Most videos go out between 07:00 and 21:00, peaking at 19:00.
Evening (18:00 to 21:00) is best both raw and within the same channel; 02:00
to 05:00 is worst. Within a channel the best hour gets about 1.9× the worst.
**Verdict: keep.** The form asks for the hour as optional, which is right.
""",
    "publish time band": """
**What it shows.** The API groups the hour into four bands. Grouping loses most
of the difference: evening is best, but the four bands are within 1.2× of each
other. **Verdict: keep, but sending the exact hour would keep more
information than the band.**
""",
    "publish weekday": """
**What it shows.** Views are almost the same on every day (0.92× to 1.06×).
It passes the rule only for the smallest channels. **Verdict: keep, weak.**
""",
    "weekend": """
**What it shows.** No difference (1.04×). **Verdict: drop.** The weekday
carries anything weekend could, and neither matters much.
""",

    # ------------------------------------------------------------ title
    "section_title": """
The title is typed on the form but is not used by the forecast today (only
for tone advice). These inputs could be worked out from it for free.
""",
    "title length": """
**What it shows.** Longer titles (over 70 characters) get slightly more views,
raw and within a channel (1.18× gap). **Verdict: keep, weak.** It repeats word
count (section 16), so one of the two is enough.
""",
    "title word count": """
**What it shows.** The same pattern as title length, a little clearer (1.27×
within a channel). **Verdict: keep instead of title length.**
""",
    "number in title": """
**What it shows.** No difference raw, but within the same channel, titles with
a number get about 1.3× those without. Numbers often mark episodes or lists.
**Verdict: keep (moderate).**
""",
    "question mark in title": """
**What it shows.** Only 6% of titles have one, and they do slightly worse.
**Verdict: keep, weak.**
""",
    "exclamation mark in title": """
**What it shows.** Only 7% of titles have one. They look much worse raw (0.64×)
but only slightly worse within a channel: the channels that use them are
smaller. **Verdict: keep, weak.**
""",
    "capital letters in title": """
**What it shows.** Raw, titles with over 30% capitals do badly (0.37×). Within
the same channel, a moderate amount (10 to 30%) is best and none is worst
(1.31× gap). **Verdict: keep (moderate).** Shouting titles belong to smaller
channels; a few capitals help.
""",

    # ------------------------------------------------------------- text
    "section_text": """
Neither is on the form today, so using them would need two new form fields.
""",
    "tag count": """
**What it shows.** 16 to 30 tags is best raw (1.33×). Within a channel, videos
with 1 to 5 tags do worst (0.71×) and 16 to 30 best, a 1.5× gap. It explains
3% on its own, more than any title input. **Verdict: keep; needs a form field.**
""",
    "description length": """
**What it shows.** Long descriptions (over 1,000 characters) get more views,
raw and within a channel (1.3× gap). **Verdict: keep; needs a form field.**
""",

    # --------------------------------------------------------- category
    "cat_profile": """
**What this shows.** Categories differ in *who* posts, not just in content.

* **News & Politics** is 31% of videos but only 57 channels: about 420 labelled
  videos each, from channels posting about 42 videos a day with a median of
  3.8M subscribers. Only 8% are Shorts.
* **Comedy, Pets & Animals, Travel & Events and Autos & Vehicles** are mostly
  Shorts (73 to 89%) from small and mid-size channels.
* **Entertainment and People & Blogs** come from very large channels (over 3M
  subscribers typical), which is much of why they rank high.
""",
    "cat_rank": """
**What this shows.** Only 188 channels post in more than one category, so the
own-channel column is based on small numbers and should be read as a hint.
It still flips the picture. **News & Politics** ranks 7th raw but 1st when a
channel that usually posts something else posts news (6.3×, 11 channels).
**Entertainment** falls from 2nd to 13th. A category's raw rank says more about
its channels than about the content.
""",
    "cat_size": """
**What this shows.** Within the same channel size, categories are much closer
than the raw ranking suggests. For example, Education is near the bottom raw,
but at 100K+ subscribers it is one of the best (8.8K to 11.9K). Howto & Style
is the worst raw category, yet 1M+ Howto channels get the highest median in
the grid. Channel size explains most of what looks like a category effect.
""",
    "grid Short or long-form": """
**What this shows.** Compared with their own channel, Shorts beat long-form in
most categories. The exceptions are **Entertainment**, where long-form wins
clearly (1.29× against 0.49×), Film & Animation and Science & Technology, where
long-form is slightly ahead, and Education, where it makes no difference. News channels' occasional Shorts do very well (2.5×).
""",
    "grid duration": """
**What this shows.** The 1 to 3 minute band is the weakest in almost every
category. Longer videos (20 minutes and more) do best in Entertainment, News,
Sports and Education; under a minute is best in Travel, Music, Gaming, Pets
and Comedy.
""",
    "grid publish time band": """
**What this shows.** Time of day matters in only a few categories:
Entertainment and Comedy do best late at night (21:00 to 23:00), News in the
evening, People & Blogs and Pets in the day. In the rest it makes no clear
difference.
""",
    "grid weekend": """
**What this shows.** Weekend against weekday makes little difference in most
categories, which agrees with section 7. **Sports** is the exception: weekend
videos get about 1.3× the channel's average, when matches are played.
""",
    "grid audio language": """
**What this shows.** Many cells are blank because few channels in a category
post in two languages. Where they do, English videos usually beat the
channel's average (1.1× to 1.3×), and "other" languages sit slightly below.
""",
    "grid number in title": """
**What this shows.** A number in the title helps most in **Entertainment**
(2.2×, often episode numbers), News & Politics and People & Blogs (about 1.2×).
It makes little difference elsewhere and slightly hurts in Music.
""",
    "grid capital letters in title": """
**What this shows.** Some capitals help in News and Entertainment; in Music,
Gaming and Travel, titles without capitals do best.
""",
    "grid title length": """
**What this shows.** Title length makes little difference in most categories.
Entertainment is the exception, where 71 to 90 characters does best (1.6×).
""",
    "grid tag count": """
**What this shows.** Many tags (31+) help in Entertainment (1.6×) and a little
in Travel, Music and Sports. Very few tags (1 to 5) are the worst choice in
Entertainment and Pets (about 0.2×) and below average in most others.
""",
    "cat_lines": """
**How to use these lines.** Each is a draft guideline for creators in that
category, based on how their videos did against their own channel. "No clear
difference" means the best and worst options were within 1.2× of each other.

Treat the extreme numbers with care. In News & Politics most channels post
long-form clips, so its rare Shorts, hour-long streams and capitalised titles
come from a small set of videos, which is why they show 2.5× to 8.5×.
""",

    # ------------------------------------------------------------ growth
    "growth": """
**What this shows.** How much a video keeps growing after its first week.

* **Science & Technology, Film & Animation, Howto & Style, Autos and Music**
  gain 13 to 21% more after day 7. These are searched for later.
* **News & Politics and Pets & Animals** gain about 1%. They are watched almost
  entirely in the first week.
* **Channels of 10K to 100K subscribers** keep growing the most (11%); very large
  channels the least (1%).
* Shorts gain slightly more than long-form (5% against 2%).

So the day-30 forecast matters most for evergreen categories and mid-size
channels; for news it is almost the same as day 7.
""",

    # ------------------------------------------------------------- thin
    "thin": """
**What this shows.** Blank or light cells are combinations the model has seen
little of. Channels under 1K subscribers in Entertainment, Sports and Science,
and 1M+ channels in Travel, Gaming, Autos, Comedy, Pets, Film and Science, are
rare or missing. A forecast for,
say, a small News channel rests on very few examples and should carry a
warning.
""",

    # ------------------------------------------------------------ drift
    "drift": """
**What this shows.** From late July on, the mix is steady: about 40% Shorts and
30% News every week, and 10,000 labelled videos a week. The first two weeks
differ (more News, fewer Shorts) because only some channels were tracked then.
Typical day-7 views dip in mid-August and rise at the start of September, from
about 950 to 1,500. A model tested on the latest weeks will therefore see
somewhat higher views than it was trained on.
""",

    # ------------------------------------------------------------ viral
    "viral": """
**What this shows.** The most-viewed 1% of videos come from a narrow group:
774 videos from 88 channels, and the top 10 channels produce 64% of them.
83% come from channels over 100K subscribers. **Entertainment is half of
them** despite being 12% of videos; **News & Politics is 31% of videos but only
4%** of the top. Shorts are slightly over-represented (50% against 40%), and so
are 20 to 60 minute videos (29% against 8%). Big errors on these few videos
will dominate any average error figure, so median-based error should be
reported alongside it.
""",

    # ---------------------------------------------------------- repeats
    "repeats": """
**What this shows.** Title length and word count, weekday and weekend, and hour
and time band each carry the same information: keep one of each pair. Short
and duration are **not** duplicates: 8,000 videos of a minute or less are not
Shorts, so the shape adds something length cannot. Channel video count and
uploads per day agree almost completely (0.94), so one of them is enough.
""",

    # ----------------------------------------------------- availability
    "availability": """
**What this shows.** Three strong inputs are available but not used:

* **YouTube topic** explains 10% of views and comes from the same channel lookup
  the API already makes.
* **Short or long-form** needs one form field.
* **Title-based inputs** can be worked out from the title the form already
  collects.

Tags and description would need new form fields.
""",

    # ---------------------------------------------------------- summary
    "summary": """
## In short

**Strongest inputs:** the channel's average views per video (38%), subscribers
(23%), the channel's YouTube topic (10%) and category (9%). Within a channel,
**duration** (2.6× gap between best and worst), **publish hour** (1.9×) and
**tag count** (1.5×) matter most.

**Drop:** channel country, captions, made for kids, HD or SD (almost all
videos the same), and weekend (no difference).

**Moderate:** audio language, number in title, capital letters, word count,
description length, time band. **Weak or only for some channel sizes:** title
script, title length, question and exclamation marks, weekday.

**Most important thing a creator controls:** Short or long-form, depending on
channel size. For channels under 100K subscribers, Shorts get about 1.35× to 2×
the channel's own long-form. For bigger channels they do not help.

**Shorts and long-form behave differently** (section 11). For Shorts, length
is the main lever (under 15 seconds about 2× better than 2 to 3 minutes). For
long-form, tags, description, a number in the title and time of day each make a
1.5× to 2.7× difference against the same channel's other videos. Guidelines in
the app should be split by format.

**For the model and the app:**

1. Fetch the channel's YouTube topic at forecast time.
2. Add a Short or long-form field to the form.
3. Use the title the form already collects to work out title inputs.
4. Consider tags and description fields.
5. Send the exact publish hour rather than a four-band time.
6. Train on the same language mapping the API uses.

**News & Politics skews the whole-dataset numbers** (section 10). It weakens
the channel-size inputs and inflates the Shorts, capital-letter and publish-hour
effects. The per-category pages are the fairer place to judge video-level
inputs, and they show that the best choices differ by category: Entertainment
and Science favour long-form, most others favour Shorts.

**Horizons.** Every input's effect is nearly the same at day 7, 14, 21 and 30
(same videos). The largest drift is subscribers, falling from 18.6% to 17.1%.
One model design across horizons is reasonable; per-horizon differences come
from growth (section 12), not from different inputs mattering.
""",
}

NOTES.update({
    "news_check": """
**What this shows.** Yes, News & Politics pulls several results, in three ways.

* **It hides how much channel size matters.** Without News, subscribers explain
  34% instead of 23%, channel video count 9% instead of 4.5%, and topic 14%
  instead of 10%. News channels are huge but get few views per clip, which
  breaks the "bigger channel, more views" pattern for everyone else.
* **It inflates some video-level effects.** News's rare Shorts, capitalised
  titles and long descriptions do very well against their own channel, so
  with News removed the own-channel gap falls for Shorts (1.17× to 1.08×),
  capital letters (1.31× to 1.10×), description length (1.30× to 1.19×) and
  publish hour (1.94× to 1.62×). Without News the best publish hour moves from
  12:00 to 21:00.
* **Raw best values flip for a few inputs:** Short or long-form (Shorts best
  with News, long-form best without), number in title and publish weekday.

**What does not change:** duration, publish time band, title length, language,
exclamation and question marks, and number in title against the own channel.
Tag count gets *stronger* without News (1.50× to 1.66×).

The Shorts finding by channel size also holds without News, and gets sharper.
Compared with their own long-form, Shorts are 2.0× for channels under 1K
subscribers and 1.4× to 1.9× up to 100K; for 1M+ channels outside News they are
0.43×.

**So:** results for one input over the whole dataset should be read with News in
mind. The per-category pages below remove the problem, because each one only
compares videos within the same category.
""",
    "inside News & Politics": """
**How to read these pages.** Inside a category, the **blue dots can still
mislead**: a handful of big channels decide most raw values (for example a
single high-volume channel that never uses tags). The **orange diamond** is the
reliable one, and the "vs own channel" columns in the table are the numbers to
quote.

**News & Politics.** Against their own channel, News videos do better when
they are Shorts (2.6×), long (60+ minutes, 8.5×), posted in the evening
(1.2×), have a number in the title (1.3×) and have some capitals or a long
description. These are the channel's *unusual* videos: most News channels post
2 to 5 minute clips, so the few streams and Shorts stand out. Tags make little
difference.
""",
    "inside Education": """
**Education.** Most inputs make little difference within a channel. The
exceptions are **length** (60+ minute videos 1.5× the channel's average, 1 to 3
minutes 0.7×) and a **long description** (3K+ characters, 1.2×). Shorts against
long-form: no difference.
""",
    "inside Entertainment": """
**Entertainment.** The category where creator choices matter most, and the one
that goes **against** the general Shorts advice: **long-form beats Shorts**
(1.3× against 0.5× the channel's average). Also better: 5 to 60 minute videos,
**late-night posting** (21:00 to 23:00, 1.7×), a **number in the title** (2.2×,
usually episode numbers), 71 to 90 character titles, **31+ tags** (1.6×; 1 to 5
tags 0.2×) and descriptions of 1K to 3K characters.
""",
    "inside Travel & Events": """
**Travel & Events.** **Shorts** beat long-form (1.2× against 0.5×), and 3 to 5
minute videos do very badly (0.3×). English audio (1.3×), many tags (31+,
1.2×) and a short or empty description do better. Time of day does not matter.
""",
    "inside Music": """
**Music.** **Shorts** beat long-form (1.2× against 0.6×). Titles **without
capitals** do best and titles over 30% capitals worst (0.7×). A number in the
title slightly hurts. Time of day makes little difference.
""",
    "inside People & Blogs": """
**People & Blogs.** Shorts do a little better (1.3× against 0.9×) and 5 to 20
minute videos best among long-form. **Avoid posting at night** (0:00 to 5:00,
0.6×). Longer titles and a number in the title help slightly.
""",
    "inside Gaming": """
**Gaming.** **Shorts** beat long-form (1.3× against 0.7×), and 5 to 20 minute
videos do worst. Titles without capitals and an empty description do slightly
better. Time of day and tags make little difference.
""",
    "inside Howto & Style": """
**Howto & Style.** Within a channel, most choices make little difference.
Shorts do slightly better (1.05× against 0.85×) and 1 to 3 minute videos worst.
The huge raw numbers on this page (for example 64× for 1 to 5 tags) come from
one or two very large channels and should be ignored.
""",
    "inside Autos & Vehicles": """
**Autos & Vehicles.** **Shorts** beat long-form (1.2× against 0.6×); among
long-form, 5 to 20 minutes is best and 1 to 3 minutes worst (0.5×). English
audio does better (1.3×). Time of day, title and tags make little difference.
""",
    "inside Sports": """
**Sports.** Long streams (**60+ minutes**, 2.9×, usually full matches) and
**Shorts** (1.2× against 0.9×) both do well; 3 to 5 minute videos worst.
Late-night posting helps a little, and so do weekends (section 10 grids).
""",
    "inside Comedy": """
**Comedy.** **Shorts** beat long-form within a channel (1.15× against 0.7×),
though the raw view says the opposite because the long-form comes from bigger
channels. **Late-night posting** is best (1.3×), evening worst. English audio
helps (1.3×), and titles with 10 to 30% capitals do worst.
""",
    "inside Pets & Animals": """
**Pets & Animals.** Almost all Shorts. The few long-form videos do very badly
(0.15× the channel's average). Posting in the day is best, at night worst.
Several gaps here are large because there are few long-form videos; treat the
exact numbers with care.
""",
    "inside Film & Animation": """
**Film & Animation.** Within a channel, almost nothing makes a difference.
Only a longer description and 5 to 20 minute videos help slightly.
""",
    "inside Science & Technology": """
**Science & Technology.** Within a channel, **long-form does slightly better
than Shorts** (1.1× against 0.9×), unlike most categories. 5 to 20 minute videos
and a 1K to 3K character description help a little; the morning is the weakest
time.
""",
})


NOTES.update({
    "format Short": """
**What this shows.** Inside Shorts, **length is the main thing a creator
controls**. Shorts under 15 seconds get about 1.4× their channel's average and
2 to 3 minute Shorts about 0.75×, roughly a 2× gap, and it holds without News.
Shorter Shorts are more likely to be watched to the end and replayed, which the
Shorts feed rewards.

Almost nothing else matters much for Shorts: time of day, weekday, a number in
the title and capital letters all stay within about 1.2×. Tamil-language Shorts
do better than the same channel's other Shorts (1.8×), but from a small number
of channels. Channel size still sets the level: Shorts from 100K to 1M
subscriber channels typically get 3.9× the typical Short.
""",
    "format long-form": """
**What this shows.** For long-form, **how the video is presented matters much
more** than for Shorts. Compared with the same channel's other long videos:

* **Length:** 60+ minutes does best (1.9×), and long-form under a minute (short
  horizontal clips) worst (0.6×).
* **Tags:** 16 to 30 tags best, 1 to 5 worst (a 1.9× gap; 2.7× without News).
* **Description:** 1K to 3K characters best, empty worst (2.0×; 1.6× without News).
* **Number in the title:** 1.3× against 0.8× (1.6× gap, with or without News).
* **Time of day:** evening best, night worst (1.5×; without News, late night
  21:00 to 23:00 is best, 1.7× gap).
* **Capital letters** looked strong (1.65×) but that is mostly News; without it
  the gap is 1.2×.

This fits how long videos are found: through search, the home page and
suggested videos, which read the title, tags and description. Shorts are found
by swiping, where those matter less.
""",
    "format compare": """
**What this shows.** The two formats rarely share the same best choice. Only
title length (71 to 90 characters) and the top category (Comedy) agree. For
tags, description, number in the title and time of day, long-form has the
larger gap and a different best value.

**For the model:** the Short or long-form input should be able to change how
other inputs act, not just shift the level. A tree model can learn this if it
is given the Short flag; a model without it would average two different
patterns. **For the app:** guidelines should be split by format. Advice on
tags, descriptions and titles belongs with long-form; for Shorts the useful
advice is mainly about length.
""",
})
