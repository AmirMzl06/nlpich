# Jigsaw روی NLP21 / CORP

این بسته برای قرار گرفتن در ریشهٔ پروژهٔ `nlpich` است؛ کنار `start_trainer.py` و پوشهٔ `utils/`.
دو فایل `jigsaw_net.py` و `mobile_jigsaw.py` عین فایل‌های جدید ارسالی شما هستند.
فایل `run_jigsaw.py` این بسته مخصوص NLP21 است و جای رانر Perich را در این پروژه می‌گیرد.
اگر رانر قبلی را لازم دارید، قبل از کپی نامش را به `run_jigsaw_perich.py` تغییر دهید.

فایل‌های لازم برای اجرا:

- `jigsaw_net.py` و `mobile_jigsaw.py`: مدل‌های ارسالی شما، بدون تغییر.
- `run_jigsaw.py`: رانر جدید NLP21.
- `nlp21_data.py`: لود trialها، پیش‌پردازش و CER.
- `nlp21_decoders.py`: دیکودرهای مستقل از encoder.
- پوشهٔ موجود `utils/` در پروژهٔ شما؛ به‌خصوص `data_loader.py`، `augmentation.py` و `constants.py`.

مدل‌ها یا trainer قدیمی پروژه import نمی‌شوند. نیازی به CEBRA، PGD یا `jigsaw_audit.py` نیست.
وابستگی‌های این مسیر `numpy`، `scipy` و `torch` هستند؛ از محیط PyTorch موجود کلاستر استفاده کنید.

## اجرای اصلی

از داخل پوشهٔ `nlpich`:

```bash
python -u run_jigsaw.py \
  --datasetPath /data/hossein/mm_project/CORP_data_release \
  --out-dir JIGSAW_NLP21_RESULTS \
  --arms order_reconstruct reconstruct_only random_encoder \
  --decoders linear_ctc gru_ctc attention_ce \
  --epochs 60 \
  --decoder-steps 20000 \
  --seeds 8
```

این دستور سه encoder/arm و برای هرکدام سه دیکودر، یعنی ۹ آزمایش downstream، اجرا می‌کند.
Jigsaw هر arm تنها یک بار آموزش می‌بیند و همان embeddingها در هر سه دیکودر استفاده می‌شوند.
حالت `random_encoder` صفر آپدیت Jigsaw دارد، ولی دیکودرهایش کامل آموزش می‌بینند.
کاهش دیکودرها به `--decoders gru_ctc` هزینه را کم می‌کند.
بودجهٔ ۶۰ epoch فقط مقدار شروع رانر است؛ ادعای بهینه‌بودن برای NLP21 نداریم.

برای مقایسهٔ ورودی خام هم `raw` را به `--arms` اضافه کنید. این حالت همان featureهای عصبی
پیش‌پردازش‌شده را با stride یکسان به دیکودر می‌دهد؛ encoder یا آموزش Jigsaw ندارد.
`raw` در این فایل به معنای بردار ویژگی هر bin است، نه flatten کردن پنجرهٔ Perich.

## دیکودرها

| نام | مدل و loss | خروجی ارزیابی |
|---|---|---|
| `linear_ctc` | لایهٔ خطی از embedding به ۳۲ کلاس؛ CTC | greedy CTC CER |
| `gru_ctc` | projection + GRU دوطرفه + لایهٔ خروجی؛ CTC | greedy CTC CER |
| `attention_ce` | GRU برای توالی ورودی + attention + GRU autoregressive؛ Cross-Entropy | CER متن تولیدشده از BOS تا EOS |

خود Jigsaw در همهٔ حالت‌ها بدون transcript آموزش می‌بیند. CTC یا CE دیکودر
فقط به دیکودر گرادیان می‌دهد. embeddingها به numpy منتقل شده‌اند و هیچ مسیر گرادیانی
به encoder باقی نمی‌ماند. این بسته fine-tuning مشترک encoder/decoder انجام نمی‌دهد.

در `attention_ce`، teacher forcing فقط در آموزش است. هنگام CER، decoder نه متن مرجع
را می‌گیرد و نه طول آن را. تکرار کاراکترها در این حالت حذف نمی‌شود. `BOS/EOS` مخصوص
این دیکودرند و در CER حساب نمی‌شوند. این مدل بزرگ‌تر و autoregressive است؛ مقایسهٔ آن
با CTC تغییر هم‌زمان دیکودر و loss است، نه آزمایش خالص اثر loss.

MSE یا CE مستقل برای هر تایم‌بین اضافه نشده: این دیتاست فقط متن trial را دارد و
alignment کاراکتر به bin در اختیارمان نیست. `attention_ce` گزینهٔ بدون CTC مناسب این قرارداد است.

## splitها و عدد نهایی

دیتالودر اصلی `utils.data_loader.get_input` استفاده می‌شود:

- `train`: همهٔ بلوک‌های هر فایل seed به‌جز بزرگ‌ترین block ID.
- `dev`: همان بزرگ‌ترین block IDهای فایل‌های seed؛ فقط برای انتخاب checkpoint دیکودر.
- `online_test`: تمام trialهای `no_recalibration` و `recalibration`؛ برای گزارش نهایی.

در جدول نهایی:

- `dev CER%`: CER checkpoint منتخب روی development.
- `online CER%`: معیار اصلی روی مجموع دو گروه online، با وزن تعداد کاراکترها.
- `project CER%`: CER روی `dev + online`؛ همان ترکیب داده‌ای که `eval_single_model.py` قدیمی
  ارزیابی می‌کرد. این عدد شامل dev مورد استفاده برای انتخاب دیکودر است و test مستقل نیست.

CER هر یک از دو گروه online نیز جدا در JSON/CSV ذخیره می‌شود. ترتیب trialهای online در
این رانر گروهی است؛ چون metric مجموع فاصله‌های ویرایشی تقسیم بر مجموع طول مرجع است،
ترتیب trialها اثری در تعریف CER ندارد. ترتیب/شمارهٔ ردیف‌ها با export قدیمی الزاماً یکسان نیست.

ارزیابی حین آموزش پروژهٔ قدیمی روی online بود؛ این رانر برای انتخاب checkpoint از dev
استفاده می‌کند. بنابراین جمعیت `project CER%` با اسکریپت نهایی قبلی یکسان است، اما
قرارداد انتخاب checkpoint تغییر کرده و باید هنگام گزارش مقایسه ذکر شود.

اینجا لیبل کاراکتر داریم، پس CER محاسبه می‌شود. PER روی این لیبل‌ها معنی phoneme error ندارد.
CER ممکن است به دلیل insertion بالاتر از ۱۰۰٪ شود؛ مقدارها clip نمی‌شوند.
مدل زبانی یا beam search به هیچ کدام اضافه نشده است.

## پیش‌پردازش و طول‌ها

1. نرمال‌سازی per-block، دقیقاً از دیتالودر فعلی، اجرا می‌شود. آمار هر بلوک از همان بلوک
   محاسبه می‌شود، حتی برای dev/online؛ این پروتکل train-only normalization نیست.
2. پیش‌فرض `--gaussian-sigma 2` است: GaussianSmoothing خود پروژه با kernel=20، داخل هر
   trial و قبل از Jigsaw، با same-padding صفر. `--gaussian-sigma 0` آن را خاموش می‌کند.
   در command قدیمی `--gauss_in` این عمل را به داخل مدل واگذار می‌کرد؛ در این رانر
   چون مدل قدیمی استفاده نمی‌شود، smoothing صریحاً انجام می‌شود. پردازش مرز هر trial
   مستقل است؛ ادعای برابری بیت‌به‌بیت با smoothing پس از padding batchِ مدل قدیمی نداریم.
3. Jigsaw لیست trialها را می‌گیرد؛ هیچ span از مرز دو trial رد نمی‌شود.
   trialهای کوتاه‌تر از `training_span` فقط از pretraining پازل کنار گذاشته و ثبت می‌شوند؛
   برای آموزش دیکودر و CER همچنان حضور دارند.
4. transform با `pad=True` روی هر trial جدا اجرا می‌شود: یک embedding برای هر bin واقعی.
   padding لبه‌ای شامل هیچ داده‌ای از trial دیگر نیست.
5. پیش‌فرض `--decoder-stride 4` است: binهای 0,4,8,... از هر توالی embedding نگه داشته می‌شوند.
   طول خروجی `ceil(T/4)` است. این stride با unfolder قدیمی kernel=32 یکسان نیست.
6. `--feature-norm train` میانگین/std embedding را فقط از train می‌گیرد و برای هر سه split
   استفاده می‌کند. `--feature-norm none` این مرحلهٔ اضافی را خاموش می‌کند.
7. در batch دیکودر، آخرین بردار تکرار می‌شود ولی طول واقعی حفظ می‌شود. GRU از packing و
   attention از ماسک استفاده می‌کنند؛ CTC هم فقط تا طول واقعی را می‌بیند.

اگر CTC طول کافی برای متن و کاراکترهای تکراری نداشته باشد، runner واضح خطا می‌دهد.
با `--decoder-stride 1` دوباره اجرا کنید. هیچ trial به‌صورت پنهانی حذف نمی‌شود و
`zero_infinity=False` است تا خطای alignment با loss صفر پوشانده نشود.

charset همان ۳۱ کاراکتر پروژه با blank=0 است. پیش‌فرض `--unknown-chars drop` رفتار حذف
کاراکتر ناشناختهٔ charset قدیمی را بازتولید می‌کند، با این تفاوت که تعداد حذف‌ها چاپ می‌شود.
برای توقف در مواجهه با کاراکتر ناشناخته، `--unknown-chars error` را بگذارید.
تبدیل خودکار space به `>` یا lowercase انجام نمی‌شود.

## بودجه و هایپرپارامترها

پیش‌فرض‌های Jigsaw: window=10، K=4، gap=(1,8)، latent=64، hidden=64، head_hidden=64،
loss weights `(order,pair,forecast,reconstruct)=(1,0.5,0,1)`، tile_norm=mean،
shuffle=True، neuron_dropout=0.1، gain_jitter=0.1، lr=1e-3.
`--levels 8` روی reconstruction نیز اثر دارد حتی وقتی forecast خاموش است.

برای کاهش حافظه، batch پیش‌فرض این رانر **۱۲۸** است؛ مقدار runner قبلی Perich **۵۱۲** بود.
با `--jigsaw-batch-size 512` قابل تغییر است. epoch یک دور کامل روی spanهای مجاز است؛
**epoch مساوی iteration نیست**. تعداد دقیق optimizer updateها ابتدای هر arm چاپ می‌شود.
دادهٔ pretraining در `fit` اصلی روی device قرار می‌گیرد؛ مقدار حافظهٔ خود ورودی نیز چاپ می‌شود.
قبل از انتخاب ۱۰۰۰۰ epoch به تعداد updateهای چاپ‌شده توجه کنید؛ تعداد trial/bin در NLP21 متفاوت است.

پیش‌فرض دیکودر GRU: hidden=256، layers=2، bidirectional، dropout=0.3، AdamW lr=1e-3،
weight_decay=1e-4، batch=16، ۲۰۰۰۰ optimizer step. هر ۵۰۰ step با dev CER checkpoint انتخاب می‌شود.
همهٔ دیکودرها بودجهٔ step یکسان دارند، اما هزینه و تعداد پارامتر یکسان ندارند.
`--max-decode-chars 512` سقف تولید attention است؛ تعداد برخورد به سقف در metrics ذخیره می‌شود.

این encoder پنجرهٔ centered می‌بیند و GRU پیش‌فرض bidirectional است: آزمایش offline است.
`--unidirectional` فقط GRU را تغییر می‌دهد و کل سیستم را causal نمی‌کند.

## MobileNet و کنترل متناظر

```bash
python -u run_jigsaw.py \
  --datasetPath /data/hossein/mm_project/CORP_data_release \
  --arms mobile_stem_mix mobile_stem_mix_random mobile_v3 mobile_v3_random \
  --decoders gru_ctc attention_ce \
  --epochs 60 --seeds 8
```

پسوند `_random` کنترل صفر epoch با همان معماری و سایر هایپرها می‌سازد.
`random_encoder` کنترل residual است و کنترل متناظر MobileNet نیست.
`--list` نام armهای موجود را نشان می‌دهد. armهای `jigsaw_audit` در این بسته ثبت نشده‌اند.
`order_reconstruct` و `proposed` در نسخهٔ فعلی alias هستند؛ یکی را انتخاب کنید.
رفتار `shuffle_tiles=True` نسخهٔ جدید را با اجرای قدیمی دارای identity labels یکی فرض نکنید.

## تست کوتاه روی کلاستر

تست مصنوعی بدون دیتاست:

```bash
python test_nlp21_pipeline.py
```

تست اتصال کل pipeline روی تعداد محدودی trial واقعی:

```bash
python -u run_jigsaw.py \
  --datasetPath /data/hossein/mm_project/CORP_data_release \
  --arms order_reconstruct random_encoder \
  --decoders linear_ctc gru_ctc attention_ce \
  --max-trials 8 --epochs 1 \
  --decoder-steps 10 --eval-every 10 --decoder-log-every 5 \
  --decoder-hidden 32 --decoder-layers 1 --seeds 8
```

`max-trials` پس از لود از ابتدای هر گروه انتخاب می‌کند؛ صرفاً برای smoke test است.
نتیجهٔ آن را به‌عنوان CER کل دیتاست گزارش نکنید. این گزینه هزینهٔ لود فایل‌ها را حذف نمی‌کند.

## استفادهٔ دوباره از encoderهای ذخیره‌شده

برای امتحان decoderهای جدید، لازم نیست Jigsaw دوباره train شود:

```bash
python -u run_jigsaw.py \
  --datasetPath /data/hossein/mm_project/CORP_data_release \
  --reuse-encoders-from JIGSAW_NLP21_RESULTS/PAST_RUN_TIMESTAMP \
  --arms order_reconstruct reconstruct_only random_encoder \
  --decoders gru_ctc --epochs 60 --seeds 8 \
  --decoder-hidden 512 --decoder-layers 3
```

تنظیم‌های Jigsaw و داده/پیش‌پردازش باید با اجرای ذخیره‌شده یکی باشند؛ runner بررسی می‌کند.
این گزینه ادامه‌دادن optimizer آموزش Jigsaw نیست؛ encoder نهایی را بازاستفاده می‌کند.
هر اجرا پوشهٔ جدید timestampدار می‌سازد. آموزش encoder با همان فایل مدل ذخیره‌شده و
دادهٔ یکسان قابل بارگذاری است؛ نسخهٔ فایل‌های مدل را بین اجراها تغییر ندهید.

## فایل‌های خروجی

- `config.json`: تنظیم‌ها، نسخهٔ کتابخانه‌ها و hash فایل‌های کد.
- `data_manifest.json`: ورودی‌ها، تعداد bin/character، splitها و پروتکل پیش‌پردازش.
- `results.json` و `summary.csv`: هر ردیف یک seed/arm/decoder با CERها.
- `pooled.json`: میانگین CER online روی seedها و sample SD در صورت وجود چند seed.
- `seed_*/ARM/encoder.pt`: encoder و headهای Jigsaw در انتهای بودجهٔ تعیین‌شده.
- `seed_*/ARM/encoder_info.json`: پارامترها، بودجه، آمار pretext و trialهای کوتاه.
- `seed_*/ARM/feature_scaler.npz`: scaler مخصوص train و stride.
- `seed_*/ARM/DECODER/decoder_best.pt`: بهترین checkpoint بر اساس dev CER.
- `metrics.json`، `history.json` و `predictions_*.json`: اعداد و متن تولیدشدهٔ هر trial.

در JSON فیلد `cer` نسبت است (0.12)، و `cer_percent` و جدول درصد هستند (12.0).
CER و آمار هر دیکودر همان موقع ذخیره می‌شوند؛ لازم نیست برای دیدنشان پایان همهٔ armها صبر کنید.

## بررسی انجام‌شده

بررسی syntax و اجرای CPU با فایل‌های MATLAB مصنوعی انجام شد. تست‌ها حفظ مرز trial،
split بلوکی، smoothing، CER تجمیعی، تکرارهای CTC، طول نامعتبر CTC، هر سه loss/decoder،
عدم تغییر encoder، تولید attention بدون لیبل مرجع، padding، ذخیره/بارگذاری و اجرای
MobileNet v2/v3 را پوشش می‌دهند. یک اجرای کامل رانر با پنج arm و هر سه دیکودر نیز گذشت.
خود دیتاست CORP و GPU کلاستر در محیط بررسی موجود نبود؛ هیچ CER واقعی برای آن ادعا نشده است.
