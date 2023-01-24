SimpleEvaluator[exp_, rules_, storage_, block_, callback_] := (

    Block[{$$$evaluated, $$$storage = <||>},
        Block[  {
                    FrontEndExecutable = Function[uid, ImportString[storage[uid], "ExpressionJSON"] ], 
                    Print = Function[x, callback@@{ToString[x/.rules, InputForm], $$$storage}]
                },

            $$$evaluated = ToExpression[exp];
            $$$out = $$$evaluated;
            If[block === True, $$$evaluated = Null];
        ];

        With[{$$$result = $$$evaluated /.rules}, 
            With[{$$$sresult = ToString[$$$result, InputForm], $$$sstorage = $$$storage, c = callback},
                CEval[ c@@{$$$sresult, $$$sstorage}; ];
            ]
        ];   
    ]);