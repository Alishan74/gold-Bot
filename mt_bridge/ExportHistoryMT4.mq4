//+------------------------------------------------------------------+
//| ExportHistoryMT4.mq4                                             |
//|                                                                    |
//| Same as ExportHistoryMT5.mq5, for MT4. Uses CopyRates/MqlRates/   |
//| TimeGMTOffset(), which are part of "New MQL4" (build 600+,        |
//| standard on any MT4 terminal from ~2014 onward) - if your broker  |
//| runs an unusually old build this may not compile; nearly every    |
//| broker's MT4 today is new enough.                                 |
//|                                                                    |
//| HOW TO RUN: same as ExportHistoryMT5.mq5 - open your gold symbol's|
//| chart, scroll back to load full history on each timeframe FIRST,  |
//| then drag this script onto the chart. Output lands in             |
//| <Terminal Data Folder>\MQL4\Files\gold_export\, same CSV format:  |
//|   line 1: # gmt_offset_seconds=<int>                              |
//|   line 2: header row                                              |
//|   line 3+: timestamp_server,open,high,low,close,volume            |
//+------------------------------------------------------------------+
#property strict
#property show_inputs

ENUM_TIMEFRAMES g_timeframes[6] = {PERIOD_M1, PERIOD_M5, PERIOD_M15, PERIOD_H1, PERIOD_H4, PERIOD_D1};
string           g_tf_names[6]   = {"1min", "5min", "15min", "1h", "4h", "1d"};

void OnStart()
{
   string symbol = Symbol();
   long gmt_offset = TimeGMTOffset();

   PrintFormat("Exporting %s history. Server GMT offset: %d seconds.", symbol, gmt_offset);

   for(int t = 0; t < 6; t++)
   {
      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      int copied = CopyRates(symbol, g_timeframes[t], 0, 100000000, rates);
      if(copied <= 0)
      {
         PrintFormat("no data for %s %s (copied=%d) - did you scroll the chart back first? skipping.",
                     symbol, g_tf_names[t], copied);
         continue;
      }

      string filename = "gold_export\\" + symbol + "_" + g_tf_names[t] + ".csv";
      int handle = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
      if(handle == INVALID_HANDLE)
      {
         PrintFormat("could not open %s for writing (error %d)", filename, GetLastError());
         continue;
      }

      FileWrite(handle, "# gmt_offset_seconds=" + IntegerToString(gmt_offset));
      FileWrite(handle, "timestamp_server", "open", "high", "low", "close", "volume");

      for(int i = copied - 1; i >= 0; i--)
      {
         FileWrite(handle,
            TimeToString(rates[i].time, TIME_DATE|TIME_SECONDS),
            DoubleToString(rates[i].open, Digits),
            DoubleToString(rates[i].high, Digits),
            DoubleToString(rates[i].low, Digits),
            DoubleToString(rates[i].close, Digits),
            IntegerToString((long)rates[i].tick_volume));
      }
      FileClose(handle);
      PrintFormat("%s %s: wrote %d bars -> %s", symbol, g_tf_names[t], copied, filename);
   }
   Print("Done. Files are under <Terminal Data Folder>\\MQL4\\Files\\gold_export\\");
}
