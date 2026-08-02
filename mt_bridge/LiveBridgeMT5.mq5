//+------------------------------------------------------------------+
//| LiveBridgeMT5.mq5 - Expert Advisor                                |
//|                                                                    |
//| Keeps the export current after the one-time backfill              |
//| (ExportHistoryMT5.mq5). Attach to ANY ONE chart of your gold      |
//| symbol and leave it running (AutoTrading must be enabled for the  |
//| EA to run, even though this never places trades). Every 30        |
//| seconds it checks each of the six tracked timeframes for newly    |
//| CLOSED bars (the currently-forming bar is always skipped - it     |
//| isn't final yet) and appends them to                              |
//| gold_export\<symbol>_<tf>_live.csv.                               |
//|                                                                    |
//| Run src/mt_import.py periodically (or wire it into                |
//| live_update.py) to pull these into data/candles/*.parquet.        |
//+------------------------------------------------------------------+
#property strict

ENUM_TIMEFRAMES g_timeframes[6] = {PERIOD_M1, PERIOD_M5, PERIOD_M15, PERIOD_H1, PERIOD_H4, PERIOD_D1};
string          g_tf_names[6]   = {"1min", "5min", "15min", "1h", "4h", "1d"};
datetime        g_last_written[6];

int OnInit()
{
   ArrayInitialize(g_last_written, 0);
   EventSetTimer(30);
   Print("LiveBridgeMT5 running for ", Symbol(), " - writing to gold_export\\", Symbol(), "_<tf>_live.csv");
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTimer()
{
   string symbol = Symbol();
   long gmt_offset = TimeGMTOffset();

   for(int t = 0; t < 6; t++)
   {
      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      // start at index 1: index 0 is the currently-forming (not yet closed) bar
      int copied = CopyRates(symbol, g_timeframes[t], 1, 5, rates);
      if(copied <= 0) continue;

      string filename = "gold_export\\" + symbol + "_" + g_tf_names[t] + "_live.csv";
      bool file_existed = FileIsExist(filename);
      int handle = FileOpen(filename, FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
      if(handle == INVALID_HANDLE) continue;
      FileSeek(handle, 0, SEEK_END);
      if(!file_existed)
      {
         FileWrite(handle, "# gmt_offset_seconds=" + IntegerToString(gmt_offset));
         FileWrite(handle, "timestamp_server", "open", "high", "low", "close", "volume");
      }

      // oldest-to-newest, only bars this EA hasn't already written this session.
      // (g_last_written resets on EA restart, which can re-write a few already-
      // seen rows into the file - harmless, mt_import.py dedupes by timestamp
      // when merging into the parquet files, same as the Dukascopy pipeline does.)
      for(int i = copied - 1; i >= 0; i--)
      {
         if(rates[i].time <= g_last_written[t]) continue;
         FileWrite(handle,
            TimeToString(rates[i].time, TIME_DATE|TIME_SECONDS),
            DoubleToString(rates[i].open, _Digits),
            DoubleToString(rates[i].high, _Digits),
            DoubleToString(rates[i].low, _Digits),
            DoubleToString(rates[i].close, _Digits),
            IntegerToString((long)rates[i].tick_volume));
         g_last_written[t] = rates[i].time;
      }
      FileClose(handle);
   }
}
