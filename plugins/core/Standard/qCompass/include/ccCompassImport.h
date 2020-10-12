#ifndef CCCOMPASSIMPORT_H
#define CCCOMPASSIMPORT_H

class QString;

class ecvMainAppInterface;


namespace ccCompassImport
{
	void importFoliations( ecvMainAppInterface *app ); //import foliation data
	void importLineations( ecvMainAppInterface *app ); //import lineation data
};

#endif // CCCOMPASSIMPORT_H
