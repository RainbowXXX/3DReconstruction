//
// Created by rainbowx on 25-5-9.
//

#ifndef UNIT_H
#define UNIT_H

struct Unit {
    auto operator<=>(const Unit&) const { return 0; }
};

#endif //UNIT_H
