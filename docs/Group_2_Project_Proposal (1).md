## In22-S5-CS3501 Data Science and Engineering Project Department of Computer Science and Engineering University of Moratuwa

# ViewCastLK

## A Data-Driven Tool for Forecasting Viewership Trends of Sri Lankan YouTube Content

## Team Members:

## AHMEDH M.R.R - 230027U AHAMED M.J.S - 230023E AHAMED M.U.A - 230025L

## Mentor: Dr. Chathuranga Hettiarachchi Teaching Assistant: Muthumala V.D.W. Group ID: 2 Project ID: P07


## 1. Executive Summary.

ViewCastLK is a data-driven forecasting tool that predicts the viewership trajectory of YouTube videos for the Sri Lankan digital content ecosystem. Existing popularity-prediction research is trained overwhelmingly on global datasets and does not capture local publishing habits, language mix, or audience behavior, leaving Sri Lankan creators, media houses and marketers without a reliable way to estimate how a video will perform before publishing.

The project collects video and channel-level data across the fifteen standard YouTube content categories using the YouTube Data API v3 as the primary source, supplemented with historical engagement data from Social Blade where available. A scheduled daily-polling pipeline tracks newly published Sri Lankan videos from their publish date onward, building the day-by-day view/like/comment history used to construct accurate labels at each forecast horizon and to derive channel-level historical performance features. The resulting time-series dataset is used to train a tree-based model that predicts view counts at multiple horizons (day 7, 14, 21 and 30), with a transfer-learning fallback and a path to sequence-based models if data volume grows sufficiently. The final deliverable is a web-based dashboard where a user can input planned video metadata and receive a forecasted viewership curve alongside actionable, data-backed publishing insights.

## 2. Problem Statement

Independent content creators, digital marketing teams and media houses in Sri Lanka currently have no localized way to estimate how a video will perform before investing time and budget into producing and promoting it. Global forecasting tools and published models are trained on datasets dominated by content from large, English-first markets and do not reflect Sri Lanka-specific factors such as bilingual/trilingual content mix, local publishing-time patterns and category preferences within a comparatively small and tightly networked viewer base.

This gap affects a growing population of Sri Lankan creators and the businesses that work with them: without data-backed guidance, content scheduling, category selection and promotional spend are largely guesswork. A tool that forecasts viewership using locally sourced data and surfaces the publishing patterns that actually correlate with performance in this market along with providing actionable insights/suggestions would let creators and marketers make evidence-based decisions instead of relying on generic international benchmarks.

## 3. Data Description

Data is sourced primarily through the YouTube Data API v3, supplemented with historical engagement data from Social Blade. Both are combined into a single time-indexed dataset spanning the project's data-collection window.

## Scope and assumption

The YouTube Data API v3 (and Social Blade) report global view counts for a video; neither provides a breakdown of viewership by viewer country. This means it is not possible to isolate “views originating from Sri Lanka” on an arbitrary video, including large Indian or international channels that Sri Lankan audiences also watch, such as global news or entertainment content. To keep the dataset well-defined and technically achievable, the project scopes its collection to videos published by Sri Lanka-based channels, on the working assumption that locally produced content - particularly in categories such as news, education, entertainment, and lifestyle is predominantly consumed by a Sri Lankan audience. This is stated explicitly as a scoping assumption and project limitation, rather than a claim that every recorded view originates from Sri Lanka.


## Primary source - YouTube Data API v3

- videos.list and playlistItems.list (1 quota unit per call) are used to enumerate a curated set of active Sri Lankan channels and pull video-level metadata cheaply.

- search.list (100 units per call, capped at 100 calls/day under the standard 10,000-unit daily quota) is reserved for filling category gaps rather than as the primary discovery mechanism, since it is the binding constraint on data collection volume.

- A scheduled daily-polling service tracks newly published videos from their publish date forward, building a genuine day-by-day view/like/comment history rather than relying on a single snapshot per video.

## Supplementary source - Social Blade

- Provides historical engagement data (views, likes, and related statistics) filterable by country, category, and made-for-kids status, which is used to extend the trajectory of videos published shortly before data collection began.

- Free-tier access constrains how far back this history reliably extends, so Social Blade data is used selectively - primarily for videos whose publish date falls within a few weeks of the point at which tracking becomes available - while the YouTube API polling pipeline remains the primary, ongoing data source.

## Coverage and scale

Data collection spans the fifteen standard YouTube content categories: Film & Animation, Autos & Vehicles, Music, Pets & Animals, Sports, Travel & Events, Gaming, People & Blogs, Comedy, Entertainment, News & Politics, Howto & Style, Education, Science & Technology, and Nonprofits & Activism. Given the project's roughly six-week data-collection window and the API constraints above, the target dataset size is on the order of several hundred to low-thousand videos with associated daily time-series records - enough to support category-level analysis while remaining realistic within quota and timeline. The exact figure will be finalized once collection is underway.

## Key features

- Video-level: publish timestamp, duration, category, title/description characteristics, made-for-kids flag.

- Channel-level: subscriber count, channel age, upload frequency.

- Time-series engagement: daily views, likes, and comment counts from publish date onward.

## 4. Methods

## Data cleaning and preprocessing

- Deduplication and handling of deleted or inaccessible videos across collection days.

- Outlier filtering (e.g. a 3-sigma rule, consistent with prior popularity-prediction literature) to remove corrupted or anomalous records.

- Timestamp normalization to Sri Lanka time for publish-time and day-of-week features.

## Exploratory data analysis

- View-trajectory shape by category, and how quickly different categories reach a stable viewership plateau.

- Effect of publish day-of-week and hour on early engagement.


- Relationship between video duration and viewership within each category.

## Feature engineering

- Channel-level historical performance - the channel's average views and engagement rate across its past videos in the same category, plus upload frequency/consistency - used as a proxy for audience reach and reliability, since this is known before the new video is ever published (unlike the video's own early engagement, which doesn't exist yet).

- Cyclical encoding of planned publish time - day-of-week and hour-of-day transformed into sine/cosine pairs so the model captures that time is circular (e.g. 11 PM and midnight are close together) rather than treating hour-of-day as a plain linear number.

- Category-normalized prediction targets (e.g. log-transformed or percentile-within-category view counts) to control for the wide scale differences between small and large Sri Lankan channels.

- Duration relative to category norm - expressing the planned video length as a deviation from that category's typical duration rather than an absolute number, since "long" means something different for a 3-minute comedy clip versus a 20-minute how-to video.

- Lightweight title-text features - length, presence of numbers or question marks, and other simple lexical signals found to correlate with performance during EDA - usable at prediction time since the title is decided before publishing.

## Forecasting model

XGBoost (gradient-boosted decision trees) is the primary model, chosen for its strong performance on structured/tabular data, training efficiency on a moderate dataset size, and interpretable feature importances a practical first baseline given the project timeline. The model predicts a view-count trajectory at day 7, 14, 21 and 30(According to data availability) from video and channel metadata available prior to publishing, enabling genuine pre-publish forecasting for the dashboard use case.

Contingency: if the Sri Lanka-specific dataset collected within the project window proves too small for robust accuracy, a model pretrained on a broader, globally sourced YouTube dataset will be fine-tuned on the Sri Lankan subset, combining global patterns with local specificity.

Scalability: as data continues to accumulate beyond the project window through the same daily-polling pipeline, retraining shifts to a sliding-window schedule (e.g. most recent 60 days) with early stopping to guard against overfitting. If data volume becomes sufficient, sequence and attention-based architectures - informed by multi- modal popularity-prediction literature - will be evaluated against the XGBoost baseline.

## System architecture

The figure below shows the end-to-end system: data ingestion from YouTube and Social Blade, a scheduled collection service, a PostgreSQL time-series store, the cleaning/feature-engineering and analysis stages, the forecasting model with its retraining loop, and the prediction API feeding the web dashboard.


## Tool development

The deliverable is a web-based dashboard (Next.js/React front end) with a metadata input form (category, duration, made-for-kids flag etc) that returns a forecasted view-count trajectory across day 7, 14, 21 and 30. Beyond the raw forecast, the dashboard translates patterns from the EDA stage into concrete, actionable suggestions tailored to the user's inputs - for example, recommending the publishing day and time associated with the strongest historical growth for the selected category, flagging when the planned video duration falls outside the range that typically performs well for that category, or suggesting an alternative category/duration/timing combination when the current inputs are forecast to underperform the category average. The dashboard also tracks historical prediction-versus-actual accuracy for transparency.

## 5. Evaluation Plan

## Model accuracy

- Mean Absolute Percentage Error (MAPE) as the primary metric, since it scores error proportionally and remains meaningful across both small and highly-viral videos.

- R² (variance explained) and Mean Absolute Error / RMSE as supporting metrics, to capture both explanatory power and the raw magnitude of prediction error.

- Comparison against a naive category-average growth-curve baseline, to demonstrate genuine predictive lift rather than accuracy that simply reflects category norms.

## Component-level evaluation

- Data pipeline: completeness and consistency checks - percentage of scheduled polls successfully collected and the missing-data rate.

- Exploratory analysis: qualitative validation against domain expectations and, where possible, cross- referenced against patterns reported in the reviewed literature.

- Forecasting model: time-based train/validation/test split (preferred over random k-fold to avoid leakage across trajectory windows).

- Dashboard/tool: informal usability check with a small group of sample users, verifying prediction latency and that the insights presented are interpretable without guidance.

Overall success is measured on two fronts: model performance (a meaningful, quantified improvement in MAPE over the naive baseline) and tool usability (a user can move from entering metadata to receiving an actionable forecast without assistance).

## 6. Expected Outcomes and Success Criteria

- A working web-based tool where a user inputs planned video metadata and receives a forecasted viewership trajectory.

- A documented, Sri Lanka-specific dataset combining YouTube API polling and Social Blade historical data, with a collection methodology that can be extended by future work.

- A trained forecasting model that outperforms a naive category-average baseline on MAPE, with the transfer-learning fallback documented if it is used.


- An analysis dashboard surfacing actionable publishing-strategy insights (category, day, and duration combinations that correlate with stronger performance) for Sri Lankan creators.

## Success criteria

- A functioning end-to-end pipeline from data collection through to the dashboard.

- Forecasting model MAPE meaningfully below the naive baseline on held-out data.

- A dashboard usable by a non-technical content creator without guidance.

## 7. Division of Work (Individual Responsibilities))

The project's work is organized around three phases - Data Collection & Engineering, Data Analysis & Modeling, and Tool Development - with tasks in each phase distributed across all three team members so that everyone contributes to every stage of the pipeline. Day-to-day task tracking is maintained separately in the team's project management system.

## Data Collection & Engineering

- Ahamed M.U.A (230025L) - YouTube Data API integration and quota-aware collection design

- Ahamed M.J.S. (230023E) - Social Blade data sourcing and integration

- Ahmedh M.R.R (230027U) -Scheduled polling service and PostgreSQL data warehouse setup

## Data Analysis & Modeling

- Ahamed M.J.S. (230023E) - Data cleaning, preprocessing, and exploratory data analysis

- Ahmedh M.R.R (230027U) - Feature engineering and XGBoost model training/evaluation

- Ahamed M.U.A (230025L) - Retraining pipeline and (if applicable) transfer-learning / sequence- model exploration

## Tool Development

- Ahamad M.U.A (230025L) - Dashboard front end and prediction API

- Ahmedh M.R.R (230027U) - Insight visualizations and historical accuracy tracking

- Ahamed M.J.S. (230023E) - Deployment and integration testing

## 8. Preliminary Bibliography

- Google Developers. YouTube Data API v3 - Getting Started. https://developers.google.com/youtube/v3/getting-started

- Xu, Y., Zheng, B., Zhu, W., Pan, H., Yao, Y., Xu, N., Liu, A., Zhang, Q., & Yan, C. (2025). SMTPD: A New Benchmark for Temporal Prediction of Social Media Popularity. arXiv:2503.04446. https://arxiv.org/abs/2503.04446

- Cho, M., Jeong, D., & Park, E. (2024). AMPS: Predicting popularity of short-form videos using multi- modal attention mechanisms in social media marketing environments. Journal of Retailing and Consumer Services, 78. https://doi.org/10.1016/j.jretconser.2024.103778

- Social Blade. https://socialblade.com - used as a supplementary source of historical YouTube engagement data.
