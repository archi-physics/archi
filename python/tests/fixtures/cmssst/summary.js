/* JavaScript */
"use strict";


var siteStatusInfo = {
   time: 1791383941,   // 2026-10-07 14:39:01
   alert: "",
   msg: "",
   url: "http://cmssst.web.cern.ch/siteStatus/",
   reload: 900
};

var siteStatusData = [
   { site: "T0_CH_CERN",
     ggus: [0, 0, 0],
     pmonth: "uuuuuuuuuuueeeeueeueuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu" +
             "uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     pweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu",
     yesterday: "uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu" +
                "uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     today: "uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T1_DE_KIT",
     ggus: [0, 0, 0],
     pmonth: "ooooowowoeeoeeeoooooowoooooooooooooooooooooooooooooooooooooo" +
             "ooowoowooowooooooooooooooooooooooooooooowooowooooooooooooooo",
     pweek: "ooooooooooooooooooooooou" + "oooooooooooooooooooooooo" +
            "oooooooooooooooooooooooo" + "oooooooooooowooooooooooo" +
            "ooooowoooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "wooooooowwooooooowwowouu",
     yesterday: "oowwwoowoooooooowwoooooowwoooowooooooooowooooooo" +
                "oooooowooowooooooooooowooooooooooooooowooowooowo",
     today: "ooooooooooooooooooooooooooooooooooooooooooooowoo" +
            "oooooooorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr",
     fweek: "rrrrrrrrrrrrrrrrrrrrrrrr" + "rrrrrrrrrrrrrrrruuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T1_ES_PIC",
     ggus: [1, 1144115, 1144115],
     pmonth: "ooooooooeoooeeoeeoooooooooooooooeooooooooooooooooooooooooooo" +
             "odddddoooowoooooooooooooooooowooooooooowoeowoooooooooooooooo",
     pweek: "ooooooouooooooooooooooou" + "ooooooooooooooooooowoooo" +
            "oooooooooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "oooooooooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "oooeeooooooooooooooooouu",
     yesterday: "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww" +
                "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww",
     today: "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwuwwwwwwwwwwwww" +
            "uwwwwwwuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T1_FR_CCIN2P3",
     ggus: [0, 0, 0],
     pmonth: "oooooowooouuueuuuuowooowwoowoooooooooowoooooooowooowooooowwo" +
             "oouuuuuuuoooooooowooooooooooooooooooowoooooooooooooooooooooo",
     pweek: "oooooooooooooooowwwwooou" + "ooooouuuuuuuuooowwoooooo" +
            "owoooowwwooowowwwwwwwwww" + "wwwwooooooueeoowoooooooo" +
            "ooowowoooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "oooooooooeweowwwwwowwwuu",
     yesterday: "ooowowowoowowooooooooooooooooooooooooowooooooooo" +
                "oooooooooooooooooooooooooooooooooooooooooooooooo",
     today: "ooooooooooooooooooooooooooooooowoooooooooooooooo" +
            "uoooouuouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T1_IT_CNAF",
     ggus: [1, 29806430, 29806430],
     pmonth: "ooooooooowoweeeooooooooowoooooooooooooooowoooooooooooooooooo" +
             "oowwoooooouuuooooooooowwowowowowwoooooooooooooowwwwwooooowwo",
     pweek: "owwwooooooooeooooooowoou" + "ooowooooooooooouuuuuuuuu" +
            "uuuuuuuppppppppwoooooooo" + "ooooowwooooooowwwwwwwoww" +
            "oowwwoowwwowoowwwwooowww" + "woooooowwwowooowowwowooo" +
            "wwwwwooowwowowowouwooouu",
     yesterday: "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwuwwww" +
                "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww",
     today: "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww" +
            "wwwwwwwwuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T1_PL_NCBJ",
     ggus: [1, 1144115, 1144115],
     pmonth: "RRRRRooooooeeTTTTTRTSRoooooooRRRRRooooooooooooeoooeooooooooo" +
             "oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo",
     pweek: "ooooowoooooeoouuuuuuuoou" + "ooooooooooooooouoooooooo" +
            "oooooooooooooooooooooooo" + "oooooooooooooeeooooooooo" +
            "ooeooooooooooooooooooooe" + "eeeeeeeeeeeeoooooooooooo" +
            "oooooooooooooooooooooouu",
     yesterday: "oowwouuuouuuouuuuuuououooouooouowwoowwwwwoowowoo" +
                "oooooooooooooooooowooooooooooooooooooooooooooooo",
     today: "oooowoooooooowwooooouoowwoooooouoooouoooouoooooo" +
            "oooouuuouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuudddduuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T1_RU_JINR",
     ggus: [1, 3717309, 3717309],
     pmonth: "oooooooooooeeeeeeeoeuoooooooooooooooooooowwoowoooowoowwowwwo" +
             "owowwowowwwwooooooooooooooooooooooooooooooooooeoooowoooooeoo",
     pweek: "oowwwowoooooooowowooooou" + "oooooooeeeeeeeeeeeeeeooo" +
            "oooooeeeeddddddoooooowoo" + "oooooeeouooooowooooeoooo" +
            "woooooooowwwwooooooooooo" + "oooooooooeoooooooooooooo" +
            "ooowoooooooooooooooooouu",
     yesterday: "oooooowwwowoooooooooooooooooooowwowooowouwwowooo" +
                "oooooooooooooooooooowwoowooowwwwoooooooooooooowo",
     today: "woowwwwweewwwwoowwooooooeeowoooooowwoooowwwooooo" +
            "wowwwououuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T1_UK_RAL",
     ggus: [0, 0, 0],
     pmonth: "ooooooooooooeooooooooooooooooooooooooooooooooooooooooooooooo" +
             "oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo",
     pweek: "ooooooooooooooooooooeoou" + "ooooooooooooooooooooooeo" +
            "eoooooooooeeoooooooooooo" + "ooooooooooooooooooooeooo" +
            "ooeooooooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "ooooooooooooeooooooooouu",
     yesterday: "woooooooooooooooooooowooowoooowooowouooooowoewwo" +
                "wwooooooooowoooooooooooooooowooooooooooooooooooo",
     today: "oooooooooowooowowwoowwwooouowwoooooooooooooooooo" +
            "oooooooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T1_US_FNAL",
     ggus: [2, 1910542, 5530986],
     pmonth: "ooooooooooooeeoooooooooooooooooooooooooooooooooooooooooooooo" +
             "ooooooooowwoooooooooooooooooooooooooooooooooooowoooooooooooo",
     pweek: "ooooooooooooooowoooowwwu" + "wooowowowooowwwwoooowoow" +
            "owoowwowwwwoowwowwwooowo" + "wowwoowwowowoowwoowwwoow" +
            "oowowooowowwwwoowoowowoo" + "oooooooooooooooooooooooo" +
            "ooooooooooooowwwooowoouu",
     yesterday: "wwwoowooooowoooooooooooooooowwoooowwwowooowwowww" +
                "ooooooooooooooowwoowowoowoooooowwooooooooooooooo",
     today: "owwoowwoowooooooooooooooooooooooooooooooooowoooo" +
            "oooooooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_AT_Vienna",
     ggus: [0, 0, 0],
     pmonth: "oowoowooooeeeeeeeeoeooooooooooowooooooooooooooooooooouuooouo" +
             "ooooweuwoooooooooooeouuooooooooooooooooooooooeoooooooooowoow",
     pweek: "eeowewoowwoowooowwoowowu" + "weooweoeoowowoewoeeeowoe" +
            "wueeuuuuuowewwoeowowwoow" + "oowwoooowwwwooooouuowooo" +
            "owwwwoooowooowwoowoooooo" + "oowowowwoowoowowooowowoo" +
            "ooowooowwooooowowooooouu",
     yesterday: "wuwwwuuwuuuwuuuwuuwwwuuwuuuwuwuuwwuwwwwwuuwwuuuu" +
                "wwwuuwwwuwuuwwuwwwwwwwwwwwuwuuwuwuwwwwwuuuuwwuww",
     today: "uwuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_BE_IIHE",
     ggus: [3, 793275, 5605914],
     pmonth: "oooooeooooeeeeeeeewoooooooowwwweoooedoowoooewoeeeeeoeooooooo" +
             "oowweoeooooweoowwooowooowwowoooooooooooooooowwwoeooooeewoooo",
     pweek: "owoeuuuuuooweewwwoweweeu" + "ewewweweewoeweewwowwwwoo" +
            "owweewwwewoewwweweewwwww" + "owewwwwwowwoooowooeeoeee" +
            "wwewewwwoowwwwoooeeoooow" + "wwwwwwoewweeeeoowooooooo" +
            "oooowowoooooooowwoooeeuu",
     yesterday: "oeeeowewwowweoweeeeweeoeoeeeeweowweeoowewewoowoo" +
                "ooewweeoweeoeeoeeeooooooooooowoooooowooooooeeeee",
     today: "eeewooooooooooooooooooooooooooooooeooowoooooooeo" +
            "ooeeoooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_BE_UCL",
     ggus: [1, 612968, 612968],
     pmonth: "ooooooooooeeeeeeeeoeoeoowwoooooooooooooooooooooooooooooooooo" +
             "ooooooooooooooooooooooooooooooowoooooooooooooooooooooooooooo",
     pweek: "ooooooeoooooooooooooooou" + "ooweooeeoooooooooooooooo" +
            "ooooooooooeooooooooooooo" + "oooooooooooooooooooooooo" +
            "oeoooooooooeeeoooooooooo" + "oooooooooooooooooooooooo" +
            "oooooooooooooooooooooouu",
     yesterday: "ooowooooooouuoooowooooooooooooooooooooowwwwoouoo" +
                "oooooooooooooooooouuooooooooooooooooooooowooowoo",
     today: "oooooowoowooooowoowoooowooooooooouooooooouwuoooo" +
            "oooooooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_BR_SPRACE",
     ggus: [0, 0, 0],
     pmonth: "ooooooooooooeeeedoooooeoooooooooooooooooooooeoeeeeeeeeweeeee" +
             "eeoooRTTVRRRRRRRRVVRRooweoowooeeeTRRTTRVVVSeooooooooooooeooo",
     pweek: "oeooooooooooooooouueeeeu" + "oooooooeoooooooooooooooo" +
            "ooooRSRSRooooooooooooooo" + "oooooooooooooooooooooooe" +
            "eeoooooooooooeoooooooooo" + "oooooooooooooooooooooooo" +
            "oooooooooooooooooooooouu",
     yesterday: "ooooooooooooooooooooooooowoooooooowooooooooooooo" +
                "oooooeoeeeooooooooeeoooooooooooooooooooooooooooo",
     today: "ooooooooooooowooooooooooooouoooooooooooooooooooo" +
            "owoooooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_BR_UERJ",
     ggus: [0, 0, 0],
     pmonth: "UUUUUUUUUUVVVVVVSVTVRSRRVVRVRSRSRSowoeeeeRoooooeeeoeeeweeoew" +
             "woeeoeowoeweewoeeeoeewwweweeeweeweewwweweeeeeeeweeweeweeewee",
     pweek: "wwweRTSTTSSSSTTTSSSSSSTV" + "TSSSSTTSTSSSSVTTSSTSSSST" +
            "TTTVTVSSTSSSSTSSTVSSSSSS" + "SSSTewwuwewwwewwweweeewe" +
            "wwewwwewwwwwwwewwwwwwwew" + "ewwewwwwwwwwwewewewweewe" +
            "ewewwwwwwwwwwewweweeweuu",
     yesterday: "wwwwuwwwowuwwwwwuwwuwwweuwwwwwwwueeeeuweeuwuwwew" +
                "eewwweeeeeeeeeeeewweueweewwwweweweeeeeeeweeeewww",
     today: "ewwwuuuuwwwwweuuuwwwuwwwuwwwuuwwwuwwwwuuwwwwwwwu" +
            "wuwuwwwwuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_CH_CERN",
     ggus: [0, 0, 0],
     pmonth: "oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo" +
             "oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo",
     pweek: "ooooooooooooooooooooooou" + "oooooooooooooooooooooooo" +
            "oooooooouuuuuuoooooooooo" + "oooooooooooooooooooooooo" +
            "oooooooooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "oooooooooooooooooooooouu",
     yesterday: "oooooooooooooooooooooooooooooooooooooooooooooooo" +
                "oooooooooooooooooooooooooooooooooooooooooooooooo",
     today: "oooooooooooooooooooooooooooooooooooooooooooooooo" +
            "oooooooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_CH_CSCS",
     ggus: [2, 3811208, 5459389],
     pmonth: "weouoweeeuoeeeeeeeweeewowooeoooowoooooweeewooooooooooooooooo" +
             "ooooooooooooooooooooooooooooooooooowoooooooooooooooooooooooo",
     pweek: "ooouoououeoooooooouuuoou" + "oooooooooeoeoooeeooooooo" +
            "ooooooooooooooooooeooooo" + "oowouuuuoooooooooooooooo" +
            "ooooeeeeeeeeeeeeeeeeeeee" + "eeeeeeeeeeeeueeueeueeeee" +
            "eeeeeeeeoooooooooooooouu",
     yesterday: "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww" +
                "wwwwwwwwwwuwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww",
     today: "wwwwwwwwwuwuuuwwwwwwwwwwwwwwuuuuuuwwuuuueeuwwuww" +
            "uwwuuuueuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_CN_Beijing",
     ggus: [3, 10842388, 53239842],
     pmonth: "oooooowoooeeeeeeeewwwoooooooowuooeoeooeoeowooewwoooooooooowo" +
             "oooooooowooooooooowoooowooooooeewoooowooooowooooooooooooowoo",
     pweek: "ewwuuuuuuwwooooeoewwwooe" + "owwoowwweewouwowowewooww" +
            "wwewoowwoeweweewwwoweooo" + "owwwwwooowowweowwowwwweo" +
            "wowowewoowooewewowooweeo" + "ooooeweewooowoeeowoowwwe" +
            "wowwoewwewowoooewooowwuu",
     yesterday: "wwoewwuououooououuuowwwuwwwuuuuwowwowwwuwwwwuwwo" +
                "uouoouuowwowoowouuwwooewooouoowwuuwwowowoouwwwou",
     today: "ouowowoewwoowowwwwwouowuoooooowwwwuoowwuwewuuuwu" +
            "ouwouuoouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_DE_DESY",
     ggus: [1, 5605454, 5605454],
     pmonth: "oeewooooooeeeeeeeeooooooooooooooooooooeooooooooooooooooooooo" +
             "oroooooooooooooooooooooooooooooooooooooooooooooooooooooooooo",
     pweek: "ooooooooooooooooooooooor" + "oooooooooooooooooooooooo" +
            "ooooooooooowoooooooooooo" + "oooooooooooooooooooooooo" +
            "oooooooooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "oooooooooooooooooooooouu",
     yesterday: "oooooooooowowoooowwooooooooooowooowwoooowooooooo" +
                "ooooooooowoooowooooooooooooooooooooooooooooooooo",
     today: "oooowooowwoowooooooowwwooooowowoooowwwwwouooowow" +
            "ooouoowouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_DE_RWTH",
     ggus: [1, 5605334, 5605334],
     pmonth: "ooooooooooeeeeeeeeoooooowooooooowooooooooooooooooooooooooooo" +
             "ooooooooooooooooooooooooooooooooooooooooooooooooooooooooudoo",
     pweek: "ooooooooooooooooooooooou" + "ooooooooooooooooooooooow" +
            "oooooooooooooooooooooooo" + "ooooooooooowoooooooooooo" +
            "oooooooooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "oooooooooooooooooooooouu",
     yesterday: "oooooooooooooooooooooooooooooooooooooooooooooooo" +
                "oooooooooooooooooooooooooooooooooooooooooooooooo",
     today: "oooooooooooooooooooooowoooooowwooooooooooooooooo" +
            "ooooowwouuuuuurrrrrrrrrrruuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_EE_Estonia",
     ggus: [0, 0, 0],
     pmonth: "ooouooooooweeeeeeeoooooooooooooooooooooooooooooooooooooooooo" +
             "ooooooeeeeewewooooooooooooooooooooooooooooooooooooooooeeeewo",
     pweek: "ooooooooooooooooooooooou" + "oooooooooooooooooooooooo" +
            "oooooooooooowooooooooooo" + "ooooooooooooowoooooooowo" +
            "oeoooooooooooooooooooooo" + "oooooooooooooooeeeeeeeee" +
            "eeeeeeeooooooooooooooouu",
     yesterday: "ooowuouowowooowouooooooooowwwooooooooooooooooooo" +
                "oowoooooooooooooooooooouooooowooooooowooooooowoo",
     today: "oooooooowooooooooooooowwwoooooowooooooooooooouoo" +
            "owwooooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_ES_CIEMAT",
     ggus: [1, 49074735, 49074735],
     pmonth: "ooooooooooooeeeeeooooooooooooooooooooooooooooooooooooooooooo" +
             "oooooooooooooooooooooooooooowooooooooooooooooooooooooooooooo",
     pweek: "ooooooooooooooooooooooou" + "oooooooooooooooooooooooo" +
            "oooowooooooowooooooowooo" + "oooooooooooooooooooooooo" +
            "oooooooooooowooooooooooo" + "oooooooooooooooooooooooo" +
            "oooooooooooooooooooooouu",
     yesterday: "oooooooooooowooooooooooooooooooooowooooooooooooo" +
                "oooooooooooooooooooooooooooooooooooooooooooooooo",
     today: "ooooooooooooooooooooooooooooooooooowoooooooeooou" +
            "uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuddduuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuudddddduuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_ES_IFCA",
     ggus: [5, 267648, 53239326],
     pmonth: "oowoooewwweeeeeeeeeeeeeeeeeeeeeeeeeeeeewooooooowooooooeowwee" +
             "ooooeeoeooeeeTTTTRRRRRRRRRRRRooeowooowoeeeeeeewowRRRRTSRSeoo",
     pweek: "oooooooooooouooooooowoou" + "oeeeeeeoowooooewoowoooew" +
            "wuoooowooowooooeeueueeeu" + "uuuuuuuueuuuuuuueuuuuuuu" +
            "uuuuVVVVVVVVVVVVVVVVVVVV" + "VVVVVVVVVTTVVVVVVVVVVVVV" +
            "VTTTTTVVVVVVVVTTTTRVRRVV",
     yesterday: "STSSVVSSVSVSVSVSSSSSSVVVSVSVVSSSVVSVVVSVSSVVSSSV" +
                "VSVVVVVSVVSSSSSVTVTTVSSSSVVVSVTSSSSVVVSVSSSVVVSS",
     today: "VSSSSVVSSSSSVSVSSSVVSVSSSSVVSSSSSSVSSSSSSVSVVSSV" +
            "VVSSVSVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_FI_HIP",
     ggus: [5, 3811207, 47711416],
     pmonth: "ooouooooooeeeeeeeeoeoooooooooooooooowooooooooooooooooooooooo" +
             "oeooooooooeeeoeeeeoooooooooooooooooooooooooooooooooooooooowo",
     pweek: "ooooooooeoooooooeoooooop" + "ooooooooooooweoooooooooo" +
            "oepooooowowooooooooooeoo" + "wooooooooooooeoeeooowwoo" +
            "ooowooooooooooooooowoooo" + "ooowoooooooooooooooooooo" +
            "ooooooooooooooooeeeeeeuu",
     yesterday: "ueeeeeewweeeeeewweewueewweeeeeeeewwwwwwwwwwwwwww" +
                "uwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww",
     today: "wwwwwwwwwwwwwwwwwwwwwwwwwwuwuwwuwwwwwuwwwwwwwwww" +
            "wuwwuwwwuuuuuuuuuuuuuupppppppppppppppppppppppppp",
     fweek: "pppppppppppppppppppppppp" + "pppppppppppppppppppppppp" +
            "pppppppppppppppppppppppp" + "pppppppppppppppppppppppp" +
            "pppppppppppppppppppppppp" + "pppppppppppppppppppppppp" +
            "pppppppppppppppppppppppp"
   },
   { site: "T2_FR_GRIF",
     ggus: [4, 92221, 6759502],
     pmonth: "oooeoeoowwweeeeeeeweeeooooooooooooooowoweoeooowoeoowewoooooo" +
             "ooooooeoowowwoeowoowoooooooooooooowoodeeeeeeeeeeeeeeeeeeeeee",
     pweek: "eeeeeeeeeeeeeeeeeeeeeeee" + "eeeeTTTTTTTVVVVVTTTTTTTT" +
            "TTTTTTTTTTTTTTTTTTTTTTTT" + "TTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTTTTTTTTTTTTTTTTT" + "TTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTTTTTTTTTTTTTTTVV",
     yesterday: "TTTTTTTTTTTTTTTTTTTTTTTTTRTTTTTTTTTTTTTTTTTTTTTT" +
                "TTTTRTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
     today: "TTTRTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTRTVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_FR_IPHC",
     ggus: [1, 5604706, 5604706],
     pmonth: "ooooooooeoeeeeeeeeoeooowowooooooooowowoooooooooooooooooooooo" +
             "oooooooooooooooooooooooooooooooooooooooooowooooooooooooouooo",
     pweek: "oooooououooooooooooeoowu" + "ooouoooooooooooooooooooo" +
            "oooooooooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "oooooooooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "oooowooooooooooooooooouu",
     yesterday: "wouuowoowouuooooooooowowowooowouowwuuoooowooowuo" +
                "wwoooowwwwwwwoooooowwowooooooooooooowwuuoooooooo",
     today: "oooowoooooooooooooooowwwoowoowwwowoooowowwowwwww" +
            "duwwowowuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_HU_Budapest",
     ggus: [1, 3811209, 3811209],
     pmonth: "ooewouooooeeeeeeeeeoowooooooooooooowoooooooooouooooooooooooo" +
             "oowoooooooooooooooooooooooooooooooooooooooooooooooooooooouoo",
     pweek: "oooooeeeeooooooooooooowu" + "owooooooowooowoooooooooo" +
            "ooooooooouuuuuooooowowoo" + "oooooooooeoewowoeooooooo" +
            "oeoooooooooooooooooooooo" + "ooooooooooooowoowoowowoo" +
            "ooowoooowwowowoooooooouu",
     yesterday: "wuuwwoowwwwwooowwoueoooooooooouooowwwwwowowwooou" +
                "uoooooooooeeooowwooooooeeeoweeoowuoowooooooooooo",
     today: "ooooooowowoowwwwwueououuwwuwuooooooooooouowwooww" +
            "wooouooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_IN_TIFR",
     ggus: [2, 3571838, 5604581],
     pmonth: "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" +
             "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
     pweek: "TTTTTTTTTTTTTTTTTTTTTTTT" + "TTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTTTTTTTTTTTTTTTTT" + "TTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTTTTTTTTTTTTTTTTT" + "TTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTTTTTTTTTTTTTTTVV",
     yesterday: "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" +
                "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
     today: "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_IT_Bari",
     ggus: [1, 12800487, 12800487],
     pmonth: "uoooouuuooeeeeeeeewoooooowoouoouooouuooooooooouoooeeeeeweeoo" +
             "eooooooooooooooooooooooooooooouoooooueeeeooooooooooooooooooo",
     pweek: "ooooooooooeooooouooooouu" + "uuuuuuooooouuuoooooooooo" +
            "oooooooooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "oooooooooouuoooooooooooo" + "owooooowoooooooooooooooo" +
            "ooooooooouuuouuuuoouuuuu",
     yesterday: "uuuuwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww" +
                "wwwwwwwwwwuuuwwwuuuuuuuuwwwwwwwwwwwwwwwwwwwwwwww",
     today: "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww" +
            "wwwwwwwuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_IT_Legnaro",
     ggus: [1, 3811207, 3811207],
     pmonth: "ooooooooooooeeoooweweeeeeeeeeeoooweeewoooooooooooooooooooooo" +
             "oooooooooooooooooooooooooooooooooooouuoooooooooooooooooooooo",
     pweek: "oooooooooooooooooooooouu" + "uuuuuuoooooooooooooooooo" +
            "oooooooooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "oooooooooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "oooooooooooooooooooooouu",
     yesterday: "oooooooooooowoooooooooooodddddwoooooodooddwdoowo" +
                "dododdwowooooodooddodooooooooooooooooooooooooooo",
     today: "oooowooooooooooooooooooooooooooooooooooooooooooo" +
            "wooooooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_IT_Pisa",
     ggus: [0, 0, 0],
     pmonth: "ooooooooooeeeTTTTTTRRRoeowowooeoooweoooooooooooooooeeeeeeeee" +
             "eeeeooooooooooooooeeoeeoeeweeeeooddddddddddddddooweeeeeeeoow",
     pweek: "ewoouowuouuuueweeeeeeeee" + "eeeeeeeeeeeooeeeeeweoeew" +
            "ewweeewwwwwowooooooowooo" + "ooowowooowwoowooooooooew" +
            "eowwweeoowewowwowowowoeo" + "owoowwoooooooooooowoowwo" +
            "oowowooowwwwowoooooowwuu",
     yesterday: "wwwuwwwwwwwwwwwwwwwwwwuwwwwwwwwwwwwwwwwwwwwwwwww" +
                "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww",
     today: "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww" +
            "wwwwwwwwuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_IT_Rome",
     ggus: [2, 268020, 3811206],
     pmonth: "ooooooooooeeeeeooooooooooooooooooooooooooooooooooooooooooooo" +
             "oooooooooooooooooooooooooooooooooooouoooooooooooooooooooowoo",
     pweek: "ooooooooooooooooooooooou" + "oooooooooooooooooooouooo" +
            "ooooooooowuuuuuuuweeweee" + "eeeeeeeeeeeeeeeeeeeeeeee" +
            "eeeeeeeeeeeeeeeeeeeeeeee" + "eeeeeeeeeeeeeeeeeeeeeeee" +
            "eeeeeeeeeeeeeeeeeeeeeeuu",
     yesterday: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" +
                "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
     today: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" +
            "eeeeeeeeuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_KR_KISTI",
     ggus: [1, 3810775, 3810775],
     pmonth: "oooooooooowoowooooooooowooooooowowoooooooooooooooooooooowooo" +
             "oeoooooooooooooooooooooooooooooowooooooooooooooooooooooooooo",
     pweek: "uoooooooooooooouuuuuuuuu" + "wowoowooooooooooooooooou" +
            "uuuuuueuoooooooooooooooo" + "oooooooooooooooooooooooo" +
            "ooooooooooooouoooooooooo" + "ooooooeooeeeeeeooooooooo" +
            "ooooooooooooooouoooooouu",
     yesterday: "ooooooooooooooooooooooooowooooooooooooooowooowow" +
                "ooouoooouuowoooooowoooooooooooooooooowouoooowwoo",
     today: "owwwwwouuowwwwwweueuwwwwwwwwwwwowwowwoowwowwwwoo" +
            "wwwoweweuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_LB_HPC4L",
     ggus: [5, 1317028, 53236300],
     pmonth: "oweeeTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" +
             "TTTTTeeeeeeeeTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTeeeeeeeeTTTTTTT",
     pweek: "TTTTTTTTTTTTTTTTTTTTTTTT" + "TTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTTTTTTTTTTTTTTTTS" + "RRTTRTSeeowwoeweeeeweewe" +
            "weeeweeeeeeweeeweeeeooww" + "eewoSSTTSSRTTTTTTSTTTTTT" +
            "TTTRTTTTTTTTRTRRRSSVSRVV",
     yesterday: "SSVSSVSSSSSSVVVVVVSSVSSVSVVVVVVSVVVSSVVVVSSVTSVV" +
                "VVVTTVSSSSSVSSVSVSVVVVSSVSVSSSSSSSVVVVSVVVSVVVVV",
     today: "VVVVVVSVVSVVVSVTSVSVVSVVVVVVSVVVSVVSSwwuuuwwuwww" +
            "wwwwwwuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_LV_HPCNET",
     ggus: [3, 5604386, 22453380],
     pmonth: "KKKKKJJJJJJJJLJJJJLJJJJJJJJJJJJJLJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJLJJLJJLJJLJJ" + "LJJLJJLJJJLJJJLJJLJJLLJJ" +
            "LJJLJJLJJLJJLJJJJJJLLJLJ" + "JJJLJJLLJJLLJJJJJLJJLJJL" +
            "LJLLJLLJLLJLLJJLJJLJJLLL",
     yesterday: "LLLLJJLLLLJJLLLJJLLLJJLLLLJJLLLLJJLLLLJJLLLLJJLL" +
                "LLJJJLLJJJLLLJJLLLJJLLLLJJLLLLJJLLLLJJLLLLJJLLLL",
     today: "JJLLLLJJLLLLJJLLLLJJLLLLJJLLLJJLLLLJJLLLJJLLLLJJ" +
            "LLLLJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_PK_NCP",
     ggus: [0, 0, 0],
     pmonth: "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" +
             "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
     pweek: "TTTTTTTTTTTTTTTTTTTTTTTT" + "TTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTTTTTTTTTTTTTTTTT" + "TTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTTTTTTTTTTTTTTTTT" + "TTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTTTTTTTTTTTTTTTVV",
     yesterday: "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" +
                "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
     today: "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_PL_Cyfronet",
     ggus: [3, 191496, 15819328],
     pmonth: "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" +
             "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJHHHIIJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "IJJJJJJJLJJJJJJJJJJLIJJJIIJJIJJJJLJJJJJJJLJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_PT_NCG_Lisbon",
     ggus: [0, 0, 0],
     pmonth: "ooooooooououeueeeeowoooooooooooooooowuowwowuooowoooowooooooo" +
             "ooooooooooooooooowoooooooooooooowooooooooooooooooooooooooooo",
     pweek: "ooowoooooooooooeooooooou" + "woeoooowooowwooeeooooooo" +
            "oooooooeoeewoooooooooooo" + "ooooeoooooooeooooowooooo" +
            "oeowooeooooooooooooooooe" + "ooooeeuouooooooooooooooo" +
            "ooooooooooeeouoooooooouu",
     yesterday: "owoowoooeewowoooouuuueeooowooooooowuuuwwwowoeewo" +
                "uwuueewooouuwoooooouwoowooooouuooowowooooooouuow",
     today: "oouooowwoooowwoooooouuuoouuoweeoouuuuuooowowooww" +
            "oooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_RU_IHEP",
     ggus: [0, 0, 0],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_RU_INR",
     ggus: [0, 0, 0],
     pmonth: "JJJJJJJJJJJJJJJJJJJJLJJJJJJJJJJJJJJJJJJJJJJJJLLJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_RU_ITEP",
     ggus: [0, 0, 0],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_RU_JINR",
     ggus: [0, 0, 0],
     pmonth: "oooooooooooeeeeeeeoewooooooooooooooooooooooooooooowooooooooo" +
             "oooooooooooooowooooooooooooooooooooooeeeoooooooooooooooooooo",
     pweek: "owoouoooooooooooooooooou" + "oooooooeoooooooouuuwoooo" +
            "oouoouoouuuuuuoooooooooo" + "ooooooooooooowooooooooeo" +
            "oooooowooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "oooooooooooooeeooooooouu",
     yesterday: "oooooooooooooowooooooooowooooooooouuoowouowooooo" +
                "ooooooooooooooooooooowowooooooooooooooooooooowoo",
     today: "wowoowwoowuowwwwoowowuuowwwouuuwowwwoooooooooooo" +
            "wowooooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_TR_METU",
     ggus: [1, 44747937, 44747937],
     pmonth: "oooooowooooeeeeeeeooooooooooooowooooowoooooooeooooooowoooooo" +
             "oooooooooooeeeewooooooooooooooooooooooooooooooooooooooooooou",
     pweek: "oooouoooooooooooooooooou" + "ooooooeeooooeoeoooooooow" +
            "oowoowooooooooooeooeoooo" + "oooooooooooooeowoooooooo" +
            "oooooooooooooooooooooooo" + "ooooweoooooooooooeeooooo" +
            "oooooooooooeoooeoooooouu",
     yesterday: "ooowooooooowwooowuowoooooooowooowooooooooooowooo" +
                "owoooowowoooooooowowooowoooooowooooowooowoowooow",
     today: "ooooowwwwwoooowwoooowooooowwoooowwwwwooowuowowow" +
            "wwwwowowuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_TW_NCHC",
     ggus: [6, 788487, 53238275],
     pmonth: "TTTTTTTTTTTTTTTTTTTTTTTTSRRSTTTTTTTTTTTSRRRTRRRRRRRRRRRRRRRR" +
             "SSoooooeeeeeeTTTTTTTTTTTRRSSRRRRSRRRSoooooooooooweeeeeeeeeee",
     pweek: "eeeeeeeeeeeeeueueeeeeeee" + "eeeeTTTTTRSSTTTTSTTTTTTV" +
            "VVVTVTTTRSSSSRSSRTSSSRSR" + "TRRRSRSSSSRSTRRSVSRSTVSS" +
            "STTTSSTSSRSSSSSSSSSSRRRS" + "TVRSooeeoooeeeeeeeeeeeee" +
            "eeeeeeeeeeeeeeeeeeeeeeuu",
     yesterday: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" +
                "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
     today: "eeeeeeeeeeeeeeeTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_UA_KIPT",
     ggus: [1, 2424882, 2424882],
     pmonth: "ddddddddddddddddddoeoeewoeooeoeeowoeowowoeowoeooowweoowoowoo" +
             "ooooowoowooeeeeowoeeeeewoooowoweoddeoooooooooooooooooooooooo",
     pweek: "uooooooeuuuuuuowuooooouu" + "ooowouooooooowoouooooooo" +
            "oooooooouuuuuuowooooowoo" + "oeooooooeeowoooooooooooo" +
            "ooooooooouoooooowooooooo" + "oooooooooooooooooouooowu" +
            "oooooououoooooooeeeeeouu",
     yesterday: "ouoouuwoowoouoowuuoowooouououowuuwuewewuuuwowwou" +
                "eeowowouuuuoooouuuuowoooooouuowuuooooouuuuuowooo",
     today: "wwuowowouuuuoouuuoowwuoooooooooooouuuwowuuouuuoo" +
            "uwoouwuouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_UK_London_Brunel",
     ggus: [0, 0, 0],
     pmonth: "oooooooooooowwowweoooooooooooooooooooooooooooooooooooooooooo" +
             "oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo",
     pweek: "oooooooooooooooooooeooou" + "oooooooowoeowwwooooooooo" +
            "ooeooooooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "oeeooooooooooooooooooeoo" + "ooooooooooooooooooooeooo" +
            "ooooooooooeooooeeooooeuu",
     yesterday: "eooeoowwoououoooouooooooooooeeooouoooooooooooooo" +
                "ooooooowowowwowowwooeeoowwooowoooooooooooooooooo",
     today: "ooowooooowooowooooowuooooooowuooooooowouooouuooo" +
            "oooooooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_UK_London_IC",
     ggus: [0, 0, 0],
     pmonth: "oooouoooooeeeeoooooooooooooooooooooooooooooooooooooooooooooo" +
             "ooooooooeeeoooooowoeeeeoooooooooooooouoooooooooooeeeoooooooo",
     pweek: "oooeoeoeeeeoeoooooooooou" + "eeeooooeuueuuueeeoeeoeoo" +
            "eoeeeeeeeeeooooooooeeeeo" + "ooeeeoeoeooeoeeeeeooeeeo" +
            "oeeeooooooeeeeeoeoeooeoe" + "eeoeeooeooeeeooeoeoeeoeo" +
            "eeeoeeoeeoeoeeeeeoeeeeuu",
     yesterday: "eeeewoooooeeooooooeeooooooooewooowueoooeoooeeeeo" +
                "oeoooowooooooooooooooooooooooooooooooooooooooooo",
     today: "oooooooooowooooooooooooooooouooooooowuuoooowoowo" +
            "oowuouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_UK_SGrid_RALPP",
     ggus: [1, 3811204, 3811204],
     pmonth: "ooouooooooeeeeooooooooooooooooooooooooooooooooooooooooweeeee" +
             "eepoooooooooooooooooooooeeeeeeeeeeeeeeeeeeoooRRRRRRRRRRRRooo",
     pweek: "oooeeoeoeowoeooeoeeeeeee" + "eeoeRTTRTRTRTTTTTTTTTTTT" +
            "TTTTTTTTTRTTTTTTTTTRTTTT" + "TTTTTTTTVTTTTTTRRTTTTTTR" +
            "RTTTRTSTTTTTTTRTTRTTTTTT" + "RTRRTRRRTSRTTTTTRTRTSTTT" +
            "TTTTSTRTTTTTTTTTTTTTTTVV",
     yesterday: "TTTVVVVVVTTTTRRRVRTTTTTTTTTTTVTTVVVVRTTVRVRTTTTT" +
                "TTTTTSVVSVVVVRRVVVRRVRRRRRVRVRRVRRRRVVVTRRRVRRVR",
     today: "VVVRRVVRVRRRRVVVVRRVRVRRVVSRRRVVRRSSRVSRVSRVVVVR" +
            "VVVRVRRVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
     fweek: "rrrrrrrrrrrrrrrrrrrrrrrr" + "rrrrrrrrrrrrrrrrrrrrrrrr" +
            "rrrrrrrrrrrrrrrrrrrrrrrr" + "rrrrrrrrrrrrrrrrrrrrrrrr" +
            "rrrrrrrrrrrrrrrrrrrrrrrr" + "rrrrrrrrrrrrrrrrrrrrrrrr" +
            "rrrrrrrrrrrrrrrrrrrrrrrr"
   },
   { site: "T2_US_Caltech",
     ggus: [1, 29455379, 29455379],
     pmonth: "ooooooooouuuuuuuuuuuuuuueuuuuuuuuuuuuVVRooowoooooooooooooooo" +
             "oooooooueoeooRRRRRRRRRRRRooooooowoooooeeooeooooooooooooooooo",
     pweek: "ooooooooooooooooooooooou" + "oowowooooooooeeeooowoooo" +
            "oooooooooooooeowwooooooo" + "ooooooooooooooeowoowoooo" +
            "ooowoooooooooeoooooooooo" + "oooowooooooooooooooooooo" +
            "oooooooooooooooooooooouu",
     yesterday: "oououooououooooooowooooooooowowooouooooooooooooo" +
                "ooooooooooowwwwooooooooooooooooooooooooooooooooo",
     today: "ooeeweeeoooooooooooooowooooooooooooooooooooooooo" +
            "oooooooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_US_Florida",
     ggus: [0, 0, 0],
     pmonth: "RRRRRSSSRoooweooooowooowoeooooooooowoooooooooeoweeeeeeeewooo" +
             "oooowoewooooooooeeeooowoooooooooooweoooooooeeoooooooooooooow",
     pweek: "eoooeooooooeeoooeoooooou" + "oooooeooooooooooooooeeeo" +
            "ooouuuooeeeeeooooooooooo" + "oooooooooooooooooooooooo" +
            "eooooooooooooooooooooeoo" + "ooooooooooooooooooooeeoo" +
            "oooooooooeeeeooooooooouu",
     yesterday: "ooooooooooooooooooooooooooooooooooooooooooooooow" +
                "woooooooooooooeeoooooooooooooooooooooooooooooooo",
     today: "eeoooooooooooooooooooooooooooooooooooeeooooooooo" +
            "oooooooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_US_MIT",
     ggus: [2, 2853793, 12733531],
     pmonth: "ooooooooooooowooooooooweeweewoeeeeoeeewooooooooooooooooooooo" +
             "ooooooooooooooooooooooeoooooooeoooooooooooooooooooooooooeueu",
     pweek: "uuueeeuuuuuuwoooooooooou" + "ooooooooooooeeeeoooooooo" +
            "oooooooooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "ooooooooooooooowoooooooo" + "ooooowoooooooooowoooooeu" +
            "uuuuuuuooooooooooooooouu",
     yesterday: "wwwwwwwwwwwwewwwwwwwuwwwwwwwwwwwwwwwwwwwwwewwwww" +
                "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwewwwww",
     today: "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww" +
            "wwwwwwwwuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_US_Nebraska",
     ggus: [2, 7415906, 9830052],
     pmonth: "owweoewooowoooooooooooooooooooweeooooooooooddddddddddddddddd" +
             "ddddddddddddddddddddddddddddddddddddddddeeeeoooooooooooooooo",
     pweek: "ooooooooooooooooooooooou" + "oooooooowooooooooooooooo" +
            "owoooooooooooooooooooooo" + "oeeeoooooooooooooooooooo" +
            "ooooooooooooeeoooooooooo" + "oooooooooooooooooooooooo" +
            "ooooooooooooooeooooooouu",
     yesterday: "uwuuueeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeewoooooo" +
                "oooooooooooooooowwuuuueeeeeeeeeeeeewoowooooooooo",
     today: "oooooooooooooooooowooooooooooooooooooooooooooooo" +
            "oouuoooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_US_Purdue",
     ggus: [2, 433380, 15819326],
     pmonth: "oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo" +
             "oooooooooooooooooooowedddddooooooooooooooooooooooooooooooooo",
     pweek: "ooooooooooooooooooooooou" + "oooooooooooooooooooooooo" +
            "oooooooooeeeeeeeeeeeeeee" + "eeeeeeeeeeeeeeeeeeeeeeee" +
            "eeeeeeeeeeeeeeeeeeeeeeee" + "eeeeeeeeeeeeeeeeeeeeeeee" +
            "eeeeeeeeeeeeeeeeeeeeeeuu",
     yesterday: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" +
                "eeeeeeeeeeeeeeeooooooooooooooooooooooooooooooooo",
     today: "oooooowowoooooooouoooowwoooooooooooooowwooooowwo" +
            "oooooowouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_US_UCSD",
     ggus: [1, 5604139, 5604139],
     pmonth: "ooooooooooooooooooooooowooooooooooooooooooooooooeeeeeeeeeooo" +
             "oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo",
     pweek: "oooooooooooooeooooooooou" + "oooooooooooooooooooooooo" +
            "oooooooooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "ooowoooooooooooooooooooo" + "oooooeoooooooooooooooooo" +
            "oooooooooooooooooooooouu",
     yesterday: "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww" +
                "wwwoowwwwwwwwwwwwwoowwwwwwwwoowwoowwwwwwwwooooww",
     today: "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww" +
            "wwwuwwwwuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_US_Vanderbilt",
     ggus: [2, 1996132, 9527298],
     pmonth: "eeweeweooooooooooooooooooooooooooooooooooooooooeooooeooeeeee" +
             "eeeeeeeeeeeeooooeoooooooooeeewooeooeeTTRRRRRRRRRRRRTTooeooee",
     pweek: "eeeeeeeeeeeeeeeeeeeuuuuu" + "ooooRRRTTTTTTSSTTTTRRTTT" +
            "TRTTTTRTTTTTTTTTTRTSTTTT" + "TTRRSRRRRRRRRRRRRSRRRRRR" +
            "RRRRRRRSRRRRRRRRRRRRRRRR" + "RRRRRRRRRRRRRRRSTTTTTTTT" +
            "RTRRooooooooooooooeeeouu",
     yesterday: "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww" +
                "wwwwwwewwwwwwwwwwwwwwwwwwwwwwweeeeewwwwwwwwwwwww",
     today: "wwwwwwwwuwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww" +
            "wwwwuuwwuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T2_US_Wisconsin",
     ggus: [1, 4527, 4527],
     pmonth: "ooeoooooooooowoooooooooooooooooooooooooooooooooooooooooooooo" +
             "oooooooooooooooooooooooooooooooooooooooooooeeeeooooooooooooo",
     pweek: "ooooooooooooooooooooooou" + "oooooooooooooooooooooooo" +
            "oooooooooooooooooooooooo" + "oooooooooooooooooooooooo" +
            "ooooooooooeooooooooooooo" + "oooooooooooeoooooooooooo" +
            "oooooooooooooooooooooouu",
     yesterday: "ooooooooouoooooooooooooooooeeeuuueeeueeuuooooooo" +
                "oooooooooooooooooooooooooooooooooooooooooooooooo",
     today: "ooooooooooooooooooooooooooooooooooeooooooooooooe" +
            "oooooooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_BG_UNI_SOFIA",
     ggus: [2, 53240038, 53240549],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_CH_CERNBOX",
     ggus: [0, 0, 0],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_CH_CERN_OpenData",
     ggus: [0, 0, 0],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_CH_PSI",
     ggus: [3, 53239967, 53240011],
     pmonth: "VVVVVVTTTVVVTTVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV" +
             "VVVVVVVVVVRVVRVVVVVVVVVVVVVVVVVVVVRTVVVTVVTVVVVVVVVVVVVVVVVV",
     pweek: "VVVVVVVVVTTVVVVVVVVVVVVV" + "VVVVVVVVRTTTTTVVVVVVVVVV" +
            "VVVVVVVVVVVVVVVVVVVVVVVV" + "VVVVVVVVVVVVVVVVVVVTTVVV" +
            "VVVVVVVVVVVVVVVVVVVVVVVV" + "VVVVVVVVVVVVVVVVVVVVVVVV" +
            "VVVVVVVVVVVVVVVVVVVVVVVV",
     yesterday: "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV" +
                "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
     today: "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV" +
            "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_CY_UCY",
     ggus: [0, 0, 0],
     pmonth: "ooooooooooooewoooooouuuuoooouuuuowoooooowooooooooooooooooooo" +
             "oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo",
     pweek: "ooowooooooooooooowoooeou" + "wouowooooouwwooooooowwoo" +
            "ooooowoowowooowooooouoww" + "oooooooooooowowoooooowow" +
            "ouoowoooouoooooooowoooow" + "wowoooowowooooowoowooooo" +
            "oooooowoooowooooooowoouu",
     yesterday: "uooouwoouuuuuouwuuououuuuououuoouuuuuouuouooooou" +
                "uuwwuuuuououuououoouuuuouuuuuoouuuwowuoooououuuu",
     today: "uuuuuououwuouuuououuuwuuuoouuuuuouuuuuuouuuwuuoo" +
            "ouuwuuuwuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_FR_IPNL",
     ggus: [2, 4425447, 53217036],
     pmonth: "RRRRRRRRRRRRRRTSRRRRRRRRRRRRRRRRRRRRRRRRTRRRRRRRRRRRRRRRRRRR" +
             "RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR",
     pweek: "RSRSSRRSRRRSRSSVSRRRRRRV" + "RRRRRRSSRSSRSSRRSSSRRSSR" +
            "RSSRRRVRVSSSRSSRRRRRRRRR" + "RRSSRRRRRRRRSRRRRRRSSRRS" +
            "RRRRRSRRRRSRSSRSRVRRRRSR" + "RRRRRSSSRSRRRSRRRRRSSRSR" +
            "RVRSRRSRRRRSRRRSRRRTSRVV",
     yesterday: "SVVVVVSVSVVVVVVSSVSSVSVVSVSVSSSSSSVVSVVVVVSSSVVV" +
                "VSSVVVSSSVVVVVSVVVVVVSSVVSVSVSVVVSVVVSSSVVSSSVVV",
     today: "VVVSVVTSVSSSVSVVVVSVSVVSVVSVSSSVVVSVSVSSSVSVSVVS" +
            "VSSVVSVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_HR_IRB",
     ggus: [0, 0, 0],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJLLLLLJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_IN_TIFRCloud",
     ggus: [0, 0, 0],
     pmonth: "RRRVRRRRRRTTTTTTTTVVVRRRRRRRVVVVVSRRRRRRRRRRRTTRRSSRRRVRRRRR" +
             "RSRRRRSRRTTRRRVVRRRRRRRRRRRRRRTVVTRRRRSRRRRRRRRRRRRRRRRRRRRS",
     pweek: "RTTTTTTRTTRTRTTTRTTRTRTT" + "RTTRTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTRTTTRRTTTTTRTTTRT" + "TTTTTRRRRRTRTTTTVTVVTVVT" +
            "TVTTVTTVTTVTTVTTVTTVTTTV" + "TVVTTVTTVTTVTTVVTVVTTVTT" +
            "VVTTVTTVTTVVTVVTTTVRRTVV",
     yesterday: "VVVSSVVVVVTSSVVSSSSSSVVVVVSTSVTSSSTTTTTTTTTTTTST" +
                "SSSSSSSSSSSSSVVVVVVVVVVVVVVVVVVVVVVVVVVVTVVVVVVV",
     today: "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVSSSVVVVVVVVVVVSSSSS" +
            "SSSSSSSSVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_IR_IPM",
     ggus: [2, 53238018, 53238235],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_IT_MIB",
     ggus: [0, 0, 0],
     pmonth: "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL" +
             "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLHLLLLLLLLLLLLLLLLLLLLLLLLL",
     pweek: "LLLLLLLLLLLLLLLLLLLLLLLL" + "LLLLLLLLLLLLLLLLLLLLLLLL" +
            "LLLLLLLLLLLLLLLLLLLLLLLL" + "LLLLLLLLLLLLLLLLLLLLLLLL" +
            "LLLLLLLLLLLLLLLLLLLLLLLL" + "LLLLLLLLLLLLLLLLLLLLLLLL" +
            "LLLLLLLLLLLLLLLLLLLLLLLL",
     yesterday: "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL" +
                "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     today: "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL" +
            "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_IT_Trieste",
     ggus: [1, 52805629, 52805629],
     pmonth: "KKKKKKKKKKKKKKJJJJLLLJLLLLLLLJJLLLLJLLLJLLJJJLJLLJLJLLJLJLJJ" +
             "JLJJJLJJLJJJLLLLLLLJLLJJLJJJJJLLJLLLLJJLJLJJLJLJJLLJJLLLLLJJ",
     pweek: "JLJJLLLLLLJJLJLJJLLJJJLL" + "JJJLJLLLLLLLJJLJLLJLLJJJ" +
            "JLJJJLLLLJJJJLJLLLJLLLLJ" + "LLLLLLLJLLLLLLLJLLLJLLLJ" +
            "JLJLLLJLJJLLJLJJJLLLLLJL" + "LJLLLLLLLJJLJLLLLLLJJLLL" +
            "LLJJLLLLLLLJJLLLLJJLJJLL",
     yesterday: "LLJJJLLLJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLJJL" +
                "LJJLJLLLLLLLJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLJLLLL",
     today: "LLLLLLLLLLLLLLLJLLLLLLLLLLLLLLJJLLLLLLLLLLLLLLLL" +
            "LLLLLJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_KR_KISTI",
     ggus: [0, 0, 0],
     pmonth: "VVVVVVVTVVVVTTTTTVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV" +
             "VTVVVVVVVVVVVVVVVVVVVVVVTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
     pweek: "TTTTTTTTTTTTTTTTTTTTTTTV" + "TVVVVVVVVVVVVVVVVVVVVVVT" +
            "VVVTTVVVVVVVVVVVVVVVVVVV" + "VVVVVVVVVVVVVTVVVVVTTVVV" +
            "TTTTVVTTTTTTTVVVVVVVVVVV" + "VVVVVVVTTVVVVVVVVVVTVVVV" +
            "TTVVVTTTVVVTTTTTTTTVVVVV",
     yesterday: "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV" +
                "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTVVVVV",
     today: "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV" +
            "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_KR_KNU",
     ggus: [0, 0, 0],
     pmonth: "LLLJLJJJJLLJJJJJJLLJLJLLJLLJLLLJLLJJJJJLLLLLLLLLLJJLLLLLJJLL" +
             "LJLLJLJLLJLLJJJJLJLJLLLLLLLLLLLLLLJJJLLJLLJLLLLLLJLLLLLLLLJJ",
     pweek: "LLLLJJJJJLLJLLJLJLLLLJJL" + "LLLLJJJLLLLLLLLLJJLLJJLL" +
            "LLLLJLLJJLLLLLLLLLJLLLJL" + "LLLLLLLLJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJLLLLLLLLLLL" + "LLLLLLLLLLLLLLLLLLLLLLLL" +
            "LLLLLLLLLLLJLLLJLLLLLLLL",
     yesterday: "LLLLLLLLLLLLLLLLLJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLL" +
                "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLJLLLLLLLLLLLL",
     today: "LLLLLLLLLLLLLLJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLJL" +
            "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_KR_UOS",
     ggus: [1, 8303067, 8303067],
     pmonth: "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV" +
             "VTVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVSVVVVVVVVVVVVVVVVVVVVVVVVV",
     pweek: "VVVVVVVVVVVVVVVVVVVVVVVV" + "VVVVVVVVVVVVTTVVVVVVVVVV" +
            "VVVVVVVVVVVVVVVVTTTTTTTT" + "TTTTTTVVVVVVVVVVVVVVVVVV" +
            "VVVVVVVVVVVVVVVVVVVVVVVV" + "VVVVVVVVVVVVVVVVVVVVVVVV" +
            "VVVVVVVVVVVVVVVVVVVVVVVV",
     yesterday: "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV" +
                "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
     today: "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV" +
            "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_MX_Cinvestav",
     ggus: [2, 53236304, 53236339],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_TW_NTU_HEP",
     ggus: [0, 0, 0],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_TW_TIDC",
     ggus: [0, 0, 0],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJLJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJLJJJJJJLJJJLJJJJJJJJ" + "JJJJJJLLJJJJJJLLJJJJJJJJ" +
            "JJJJJJJJLJJJJJJJLJJJJJJJ" + "LJJJJJLJJJLJJJJJJJJJJJJJ" +
            "JLJJJJJJJJJLJJJJJJLJJJJJ" + "JLJJJJJJJJJLJJJJJJJJJLJJ" +
            "JLJJLJJJJJJLJJJJJJJJLJLL",
     yesterday: "JJLLLLJJLJLJJLLLJJJLLJJJLLLJJJJLLJJJLLJJJLLJJJLL" +
                "LJJJLLJJJLLLJJJJLLJJJLLLJJJJJLJJJLLLJJJLLJJJLLLJ",
     today: "JJLLLJJJLJLJJJLLLJJJLLLJJJLLLJJJLLJJJJLLLJJJLLLJ" +
            "JLLLLJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_UK_London_QMUL",
     ggus: [0, 0, 0],
     pmonth: "eeooooowoeeeeweeeeweueoeeewueeeeeeueuooeeeeeeewwowewuuuuuuuu" +
             "uuuuuuuuuuweewoewowooeowooowewuuuwweoooooooeoooowwoooooooooo",
     pweek: "eoeeeoeeeeoeeoeooeeeoeee" + "oeeeeeeeeoeeeeeeeeeeeeee" +
            "eeeeeeeeeeeeeeeeeeueeeee" + "eeueeeeeuueueeoeeeeeeeee" +
            "eeeeeeoeeeeeeeeeeeeooeee" + "eeeeoeeeoooooooeeeeeeeee" +
            "eoeoeeeeeeeeeeeeeeeeeeuu",
     yesterday: "eeeoooooeeeeeeeeeeeooueeeeeueeueeeeueeeeeeeeeeeu" +
                "eeeeooooeeeeeeooooooooooooooooooooooooooouueoouo",
     today: "oooeeooooooooooooooooooooeoouoooooooooouoouooooo" +
            "oeoooooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_UK_London_RHUL",
     ggus: [0, 0, 0],
     pmonth: "ooouoooouuwoweeeeeowwueuuoouooouooowoououoooououuuwooouoooou" +
             "oouoouuooowoooouuuuuuuuuuuuuuuuuuuuuuwwuuueouuuoouuuoooooooo",
     pweek: "ooeoeeeeeeeeeoouoeeoeeee" + "eeeeeeeeeeeeeeoeeeeeeeee" +
            "eeeeeeeueouuueeeueuuueee" + "ooeeeeeeeeeeeoeoeeuuueeu" +
            "uuuuueeeeeeeeeouoeeeeuuu" + "uoooeeoeeuuoeoeeeeoeeeee" +
            "eeeeoeeeeeeeeeueeeeoeeuu",
     yesterday: "ooeeeeooeeeeeeeeooooeeeeeeeeooeeeeooeeeeeeeeoooo" +
                "oueeooeeeeeeeeeuooooooooooooouuuuuuuuuoowwoooooo",
     today: "oowwouooooouuoooowwwwoooowwoooouuooooooooooooooo" +
            "oooooooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_UK_SGrid_Bristol",
     ggus: [1, 26532614, 26532614],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_UK_SGrid_Oxford",
     ggus: [0, 0, 0],
     pmonth: "ooooooowweweeeeeeewoeoowwewwooweeoewwewoweoeeowoewooeoowwoow" +
             "ooooowoeoooeooowowweeooeeewuuuuuuuuuuueouuuuuwooeowouuuuuuuu",
     pweek: "uuuuuuuuuuuuuuuuuoeouuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuueuuoueuuuuuu" + "uuuuuuuuuuuuuueuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuueuuuuuouuuuuuuueeuu" +
            "uuuuuuueeuuuuuueuuuuuuuu",
     yesterday: "uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuueeeuuueee" +
                "eeeoooooooooooouooooooeeooeeooeeeuwweeeeeuoooooo",
     today: "eewweeoooooooowwoowwwwwwwwoooooeeeewwoowwooeewwo" +
            "ooooowwwuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_UK_ScotGrid_GLA",
     ggus: [0, 0, 0],
     pmonth: "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTSTTTTTTRRRSRRRRRRRRRRRooooooo" +
             "oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo",
     pweek: "oooeeeeeoouooooeoooooeeo" + "ooeeeooeoeeooueeooeeeooo" +
            "oooeoeeoeoeeooooeeeoooee" + "oeeeeooooeooeoeeeeoeeeoo" +
            "oeeeoooeoeooooeoeeeoeooe" + "eeeeeoooooeooooooooeeeeo" +
            "eooeeoeeeoeeeeeeeeeeeeuu",
     yesterday: "eeoooooooooooeeeeeeeooeeeeeeeeooeeeooeeooeeeeeee" +
                "oooewooeeeeeeeoooooooooooooooooooooooooooooooooo",
     today: "oooooooooooooooooooooooooooooooooooooooooooooooo" +
            "oooooooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_US_Baylor",
     ggus: [1, 53236259, 53236259],
     pmonth: "TTTTTTTTTTTTTTTTRSRRTRRTRRTRSRRRRRRSTTTRSSTTRRTRRTRSRRTRRRRS" +
             "RRRRRRRRRRSTRRRRTRRRRRRRRRRRRRRRSRSRRRRRRRRRRRRTRRRRRRRRVRRR",
     pweek: "SRSSSRRRSVVVTVVSSRSRSRSV" + "VVVVSSRSSRSSSSSSRSSRSSSR" +
            "RSRSSSRRRSSSRSRSRSSSSSRS" + "SRSRSRRSSSRRSSRSSRSSRSSS" +
            "SRSSSRSSRSRRSRRRSRRSRRSR" + "SRSSRTRVRSSSSSSRRSRRSRRR" +
            "RRSSSSRSRRSRSSSRSRSSRSVV",
     yesterday: "SVSSSVVSVSVSVSSVVSVVVVVVVSSVSVVVSVVVSSSVSSVSSSVV" +
                "SSSVSVSSSSVSSSSSSVSVVVVSVSVSVSVVVVVSSVSSSSVVSSSS",
     today: "SVVSSVVSSSSVVSSSVSSSSSVSSSSSSSVSSVSVSSSVSVSSSVSS" +
            "VVSVSSSSVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_US_Brown",
     ggus: [4, 3021039, 53236367],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_US_CMU",
     ggus: [0, 0, 0],
     pmonth: "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" +
             "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
     pweek: "TTTTTTTTTTTTTTTTTTTTTTTT" + "TTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTTTTTTTTTTTTTTTTT" + "TTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTTTTTTTTTTTTTTTTT" + "TTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTTTTTTTTTTTTTTTVV",
     yesterday: "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" +
                "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
     today: "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_US_Colorado",
     ggus: [0, 0, 0],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_US_FNALLPC",
     ggus: [0, 0, 0],
     pmonth: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" +
             "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
     pweek: "eeeeeeeeeeeeeeeeeeeeeeee" + "eeeeeeeeeeeeeeeeeeeeeeee" +
            "eeeeeeeeeeeeeeeeeeeeeeee" + "eeeeeeeeeeeeeeeeeeeeeeee" +
            "eeeeeeeeeeeeeeeeeeeeeeee" + "eeeeeeeeeeeeeeeeeeeeeeee" +
            "eeeeeeeeeeeeeeeeeeeeeeuu",
     yesterday: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" +
                "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
     today: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" +
            "eeeeeeeeuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_US_MIT",
     ggus: [0, 0, 0],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_US_NotreDame",
     ggus: [1, 53236323, 53236323],
     pmonth: "JJJJLLLLLLLLLJLLLLLLLLLLLLLLLLJLLLLLLLJLLLLLLLLLLLLLLLLLLLLL" +
             "LLLJLLLLJLLLLLLLLLJJLLLLLLJLLLLLLLLLLJLLLJLLJJJJJJJJJJJLLLLL",
     pweek: "LLLLJLLLLLLLLLLLLLLLJLLL" + "LLLLLLLLLLLLLJJJJJLLLLJJ" +
            "JJLLLLLLLLLJLLLLLLLLLLLL" + "LLLLLLLJLLLLLJJLLLLLLLLL" +
            "LLLJLLLJJLLLLLLLLLLLLLLL" + "JJLLJJLLJJLLJJJLLLLLLLLL" +
            "LJLLJLLLLLLJLLLJLLLJLJLL",
     yesterday: "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL" +
                "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLJJLLLLLLLLL",
     today: "LLLLLLLLLLLLLLLLJJLLLLLLJJLLLLLLLLLLLLLLLLLLLLLL" +
            "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_US_OSG",
     ggus: [1, 11514097, 11514097],
     pmonth: "JJJJJJJJLLLLLLLJLLLLLLLLJJLLLLLLLLLLLLLLLJJJJLJJJJLLLLLJLLLJ" +
             "LLJJLLLLJJLLLLLLLLLLLLLJLLLLJJJLLLJLLJLLJLJJJLJJJLLLLLLLLLLL",
     pweek: "LJLLLLLLLLLLLLLLLLLLLLLL" + "JLLLLLLLLLLLLLLLLLLLLJLL" +
            "JJLLLLLLLLLJJLLLLJJJLLLJ" + "LLLLLLLJLLLLLLLJLLLLJLLL" +
            "LLLLLLLLLJLLLLLLLLLLJLLL" + "LLLJLLLJLLLJLLLLLLJJLLLL" +
            "LLLLLLLJJJLLLLLLLJLLLLLL",
     yesterday: "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLJJJ" +
                "JJJJJJJLLJJLLLLLLLLLLLLLLLLLLLLLLLLLJLLLLJJLLLLL",
     today: "LLLLLLLLJJJJJJLLLLLLLLJJJJLLLLLLLLLLLLJJLLLLLLLL" +
            "LLLLJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_US_Princeton_ICSE",
     ggus: [0, 0, 0],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_US_PuertoRico",
     ggus: [1, 27996009, 27996009],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_US_Rice",
     ggus: [0, 0, 0],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJJJ" + "JJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJJJJJJJJJJJJJJJLL",
     yesterday: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
                "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     today: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
            "JJJJJJJJLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_US_Rutgers",
     ggus: [0, 0, 0],
     pmonth: "owouoooewowwoeeoowooeeeoooeeoooeeewwoooewwoeoweooeeoooooooew" +
             "eoeeoTTTTTTTRTRTRTTTRRRRRRTSTRSRSeoeowooooooooeooowooooooooo",
     pweek: "wooooooowwooooowwuuuuuuu" + "oowoooowowooowowoooooooo" +
            "ooowoooooooowooooooooouo" + "owwwowwowooouowowoouowoo" +
            "oooooooowoeeoooooowooooo" + "woooooowwwoooowweeooooww" +
            "wwooooowoooooooooooooouu",
     yesterday: "wwuuuwuwuuuwuwuwuuwwwuuwwwwuuwwwuwwuwwwuuwwuuwww" +
                "wuwuwwwuuuuwuuuwuueeeueewwwuuuuwuuwuwuwuwwwwuuwu",
     today: "uuuwuuuwwwwwuuwuwwwuuuuuuuuuwwuuuuuwwwuwuuwuwuuw" +
            "uwwwuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_US_UMD",
     ggus: [2, 53236257, 53236262],
     pmonth: "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ" +
             "JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ",
     pweek: "JJLLJJJJJJJJJJJJJJJLLJJL" + "JJJLJLJJJJJLLJJJJJJJLJJJ" +
            "JJJJLLLJJJJJJLLJJJJJJJLJ" + "JJJJJLLLJJJJJJLLJJJJJJLL" +
            "LJJJJJJLJJJJJJJJJJJJJJJJ" + "LLJJJJJJLLJJJJJJJLJJJJJJ" +
            "JLLJJJJJJJLJJJJJJJLLJJLL",
     yesterday: "JJLLJJJJLLLLLLLLLJJLJJJLJJJLLLLLJJJLJLJLLLLLLLLL" +
                "LLLJLJLJLJLJJLLLLLLJLJLJLLLLLLLLLLLLJLJLJLJLJLLL",
     today: "LLLLLJLJLJLJJJLLLLLLLJJLLLLLJLJJLLLLLLJLJLJLLLJL" +
            "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   },
   { site: "T3_US_UMiss",
     ggus: [1, 3818952, 3818952],
     pmonth: "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" +
             "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
     pweek: "TTTTTTTTTTTTTTTTTTTTTTTT" + "TTTTTTTTTTTTTTTTTTTTTTTT" +
            "TTTTTTTTTTTTTTTTTTTTTTTT" + "TTTTTTTTTTTTTTTTTTTVVVVV" +
            "VVVVVVVVVVVVVVVVTVVVVVVV" + "VVVVVVVVVVVVVVVVVVVVVVVT" +
            "VVTTTTTTTTTTTTTTTTTTTTVV",
     yesterday: "VVVTVVVTVVVTVTVTVTVTVVVVVTTVVVVVTVTVTVVVTVVVTVVV" +
                "TVVVVTVVVVTTVVVVVTVTVTVVVTTTTVTVTVTVTVTVVVTVTTVV",
     today: "VVVTVTTVVVVTVVVTVTVTVTVVVVVVTVTVVVVVTVTVTVVVTVVV" +
            "TVVTTTVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
     fweek: "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu" + "uuuuuuuuuuuuuuuuuuuuuuuu" +
            "uuuuuuuuuuuuuuuuuuuuuuuu"
   }
]