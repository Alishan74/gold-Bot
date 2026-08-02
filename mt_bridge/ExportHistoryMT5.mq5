//+------------------------------------------------------------------+
//| ExportHistoryMT5.mq5                                             |
//|                                                                    |
//| One-time (re-runnable) historical backfill: exports full available|
//| OHLCV history for the CURRENT chart's symbol, across the six      |
//| timeframes this project uses, to CSV files under                 |
//| <Terminal Data Folder>/MQL5/Files/gold_export/.                  |
//|                                                                    |
//| HOW TO RUN:                                                       |
//| 1. Open a chart for your broker's gold symbol (whatever it's      |
//|    actually called - XAUUSD, XAUUSD.a, GOLD, etc. - this script   |
//|    uses Symbol() so it exports whatever chart you drop it on).    |
//| 2. IMPORTANT: MT5 only has locally what's been downloaded from    |
//|    the broker's server. Before running this, press End then hold  |
//|    Home (or scroll back / press the Home key repeatedly) on EACH  |
//|    timeframe you care about until the chart stops loading older   |
//|    bars, so the terminal actually has that history cached first.  |
//|    Skipping this step will silently give you less history than    |
//|    your broker actually has.                                      |
//| 3. Drag this script onto the chart (Navigator > Scripts).         |
//| 4. Find the output in File > Open Data Folder > MQL5 > Files >    |
//|    gold_export\                                                   |
//|                                                                    |
//| Output format (one file per timeframe, <symbol>_<tf>.csv):        |
//|   line 1: # gmt_offset_seconds=<int>   (broker server time minus  |
//|           GMT, in seconds, from TimeGMTOffset() - includes server |
//|           DST if your broker's server observes it)                |
//|   line 2: header row                                              |
//|   line 3+: timestamp_server,open,high,low,close,volume            |
//|            oldest bar first                                       |
//+------------------------------------------------------------------+
#property script_show_inputs
#property strict

ENUM_TIMEFRAMES g_timeframes[6] = {PERIOD_M1, PERIOD_M5, PERIOD_M15, PERIOD_H1, PERIOD_H4, PERIOD_D1};
string          g_tf_names[6]   = {"1min", "5min", "15min", "1h", "4h", "1d"};

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

      // rates[] is newest-first (series order) - write oldest-first for a clean chronological file
      for(int i = copied - 1; i >= 0; i--)
      {
         FileWrite(handle,
            TimeToString(rates[i].time, TIME_DATE|TIME_SECONDS),
            DoubleToString(rates[i].open, _Digits),
            DoubleToString(rates[i].high, _Digits),
            DoubleToString(rates[i].low, _Digits),
            DoubleToString(rates[i].close, _Digits),
            IntegerToString((long)rates[i].tick_volume));
      }
      FileClose(handle);
      PrintFormat("%s %s: wrote %d bars -> %s", symbol, g_tf_names[t], copied, filename);
   }
   Print("Done. Files are under <Terminal Data Folder>\\MQL5\\Files\\gold_export\\");
}
