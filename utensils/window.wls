Options[CreatePage] = {"id":>CreateUUID[], "postHandler"->Function[{x,y}, "index.wsp"], "eventHandler"->Print, "postRedirect"->Function[data, ("index.wsp")]};

CreatePage[content_, OptionsPattern[]] := Module[{},
    console["log", " --- page generated with id ``", OptionValue["id"]];
    console["log", " --- page generated with id ``", OptionValue["id"]];
    window[OptionValue["id"]] = 
        <|  "content"->content,
            "eventHandler"->OptionValue["eventHandler"],
            "postHandler"->OptionValue["postHandler"],
            "postRedirect"->OptionValue["postRedirect"]

        |>;
    
    OptionValue["id"]
];

PageEmitt[id_][message_] := window[id, "eventHandler"]@@{client, message};