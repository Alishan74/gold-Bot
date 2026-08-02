//+------------------------------------------------------------------+
//| LiveBridgeMT4.mq4 - Expert Advisor                                |
//|                                                                    |
//| MT4 equivalent of LiveBridgeMT5.mq5 - see that file for the full  |
//| explanation. Attach to any one chart of your gold symbol, enable  |
//| AutoTrading (this EA never places trades, but MT4 requires        |
//| AutoTrading on for any EA's OnTimer to run), and leave it running.|
//+------------------------------------------------------------------+
#property strict

ENUM_TIMEFRAMES g_timeframes[6] = {PERIOD_M1, PERIOD_M5, PERIOD_M15, PERIOD_H1, PERIOD_H4, PERIOD_D1};
string           g_tf_names[6]   = {"1min", "5min", "15min", "1h", "4h", "1d"};
datetime         g_last_written[6];

int OnInit()
{
   ArrayInitialize(g_last_written, 0);
   EventSetTimer(30);
   Print("LiveBridgeMT4 running for ", Symbol(), " - writing to gold_export\\", Symbol(), "_<tf>_live.csv");
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

      for(int i = copied - 1; i >= 0; i--)
      {
         if(rates[i].time <= g_last_written[t]) continue;
         FileWrite(handle,
            TimeToString(rates[i].time, TIME_DATE|TIME_SECONDS),
            DoubleToString(rates[i].open, Digits),
            DoubleToString(rates[i].high, Digits),
            DoubleToString(rates[i].low, Digits),
            DoubleToString(rates[i].close, Digits),
            IntegerToString((long)rates[i].tick_volume));
         g_last_written[t] = rates[i].time;
      }
      FileClose(handle);
   }
}
