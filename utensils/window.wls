Options[CreatePage] = {"id":>CreateUUID[], "postHandler"->Function[{x,y}, "index.wsp"], "eventHandler"->Print, "postRedirect"->Function[data, ("index.wsp")]};
Options[PagePlaceholder] = {"id":>CreateUUID[], "postHandler"->Function[{x,y}, "index.wsp"], "eventHandler"->Print, "postRedirect"->Function[data, ("index.wsp")]};

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

PagePlaceholder[id_, OptionsPattern[]] := (
    Tinyweb`WebSocketPublish[server, LoadDynamicPage[PageLoadingCached], id];
    window[id] = 
        <|  "content"-> PageLoadingCached,
            "eventHandler"->OptionValue["eventHandler"],
            "postHandler"->OptionValue["postHandler"],
            "postRedirect"->OptionValue["postRedirect"]

        |>;
);

DeletePage[id_] := (window[id]=.; Tinyweb`WebSocketPublish[server, PageRedirect["index.wsp"], id]);

PageLoadingCached = LoadPage["assets/loading.wsp", {}, "base"->$env["path"] ];

PageEmitt[id_][message_] := window[id, "eventHandler"]@@{client, message};