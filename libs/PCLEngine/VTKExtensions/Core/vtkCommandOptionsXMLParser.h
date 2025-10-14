// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef vtkCommandOptionsXMLParser_h
#define vtkCommandOptionsXMLParser_h

#include "qPCL.h"               // needed for export macro
#include "vtkCommandOptions.h"  // for enum
#include "vtkXMLParser.h"

class vtkCommandOptionsXMLParserInternal;
class vtkCommandOptions;

class QPCL_ENGINE_LIB_API vtkCommandOptionsXMLParser : public vtkXMLParser {
public:
    static vtkCommandOptionsXMLParser* New();
    vtkTypeMacro(vtkCommandOptionsXMLParser, vtkXMLParser);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    //@{
    /**
     * Add arguments to the xml parser.  These should be the
     * long arguments from the vtkCommandOptions class of the form
     * --foo, and pass in the variable that needs to be set with the value.
     */
    void AddBooleanArgument(const char* longarg, int* var, int type = 0);
    void AddArgument(const char* longarg, int* var, int type = 0);
    void AddArgument(const char* longarg, char** var, int type = 0);
    void SetPVOptions(vtkCommandOptions* o) { this->PVOptions = o; }

protected:
    //@}
    /**
     * Default constructor.
     */
    vtkCommandOptionsXMLParser();

    /**
     * Destructor.
     */
    ~vtkCommandOptionsXMLParser() override;

    // Called when a new element is opened in the XML source.  Should be
    // replaced by subclasses to handle each element.
    //  name = Name of new element.
    //  atts = Null-terminated array of attribute name/value pairs.
    //         Even indices are attribute names, and odd indices are values.
    void StartElement(const char* name, const char** atts) override;

    // Called at the end of an element in the XML source opened when
    // StartElement was called.
    void EndElement(const char* name) override;
    // Call to process the .. of  <Option>...</>
    void HandleOption(const char** atts);
    // Call to process the .. of  <Option>...</>
    void HandleProcessType(const char** atts);

    virtual void SetProcessType(const char* ptype);
    void SetProcessTypeInt(int ptype);

private:
    vtkCommandOptionsXMLParser(const vtkCommandOptionsXMLParser&) = delete;
    void operator=(const vtkCommandOptionsXMLParser&) = delete;
    int InPVXTag;
    vtkCommandOptions* PVOptions;
    vtkCommandOptionsXMLParserInternal* Internals;
};

#endif  // #ifndef vtkCommandOptionsXMLParser_h
