Getting Support
===============

We're here to help! This page outlines the various channels for getting support with ACloudViewer.

Documentation
-------------

The first place to look for answers:

📚 **Official Documentation**

- **Getting Started**: :doc:`/getting_started/introduction`
- **Tutorials**: :doc:`/tutorial/index`
- **Python API**: :doc:`/python_api/cloudViewer.core`
- **C++ API**: :doc:`/cpp_api`
- **FAQ**: :doc:`faq`

🔍 **Search the Documentation**

Use the search box at the top of this page to find specific topics.

Community Support
-----------------

GitHub Discussions
^^^^^^^^^^^^^^^^^^

The best place to ask questions and get help from the community:

💬 `GitHub Discussions <https://github.com/Asher-1/ACloudViewer/discussions>`_

**Categories:**

- **Q&A**: Ask technical questions
- **Show and Tell**: Share your projects
- **Ideas**: Propose new features
- **General**: General discussions

GitHub Issues
^^^^^^^^^^^^^

For bug reports and feature requests:

🐛 `GitHub Issues <https://github.com/Asher-1/ACloudViewer/issues>`_

**Before creating an issue:**

1. Search existing issues
2. Check if it's already fixed in the latest version
3. Provide minimal reproduction steps

**Issue Template:**

.. code-block:: text

   **Describe the bug**
   A clear description of what the bug is.
   
   **To Reproduce**
   Steps to reproduce:
   1. Load point cloud...
   2. Apply filter...
   3. See error
   
   **Expected behavior**
   What you expected to happen.
   
   **Environment:**
   - OS: [e.g., Ubuntu 22.04]
   - ACloudViewer version: [e.g., 3.9.3]
   - Python version: [e.g., 3.11]
   - CUDA version: [e.g., 12.1]
   
   **Additional context**
   Any other relevant information.

Stack Overflow
^^^^^^^^^^^^^^

Ask questions with the ``acloudviewer`` tag:

📖 `Stack Overflow <https://stackoverflow.com/questions/tagged/acloudviewer>`_

Good for:

- Algorithm questions
- Integration with other libraries
- General 3D processing questions

Email Support
-------------

For sensitive issues that cannot be discussed publicly:

📧 **Email**: support@acloudviewer.org

Please note that response times may vary, and community channels are generally faster.

Professional Support
--------------------

Commercial Support
^^^^^^^^^^^^^^^^^^

For organizations requiring:

- Priority bug fixes
- Custom feature development
- Training and consultation
- Long-term support contracts

Contact: enterprise@acloudviewer.org

Training & Workshops
^^^^^^^^^^^^^^^^^^^^

We offer:

- Online training courses
- On-site workshops
- Custom training for teams

For inquiries: training@acloudviewer.org

Financial Support
-----------------

Support ACloudViewer Development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ACloudViewer is an open-source project maintained by volunteers. If you find it useful and want to support its development, consider making a donation.

**Why Donate?**

- Help fund ongoing development and maintenance
- Support new feature development
- Enable better documentation and tutorials
- Help us provide community support

**How to Donate:**

.. raw:: html

   <div style="text-align: center; margin: 30px 0;">
      <p style="font-size: 16px; margin-bottom: 20px;">Scan the QR code below to donate via WeChat Pay:</p>
      <img src="../_static/donation.png" alt="Donation QR Code" style="max-width: 300px; border: 2px solid #ddd; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
      <p style="font-size: 14px; color: #666; margin-top: 15px; font-style: italic;">
         Thank you for your support! 🙏
      </p>
   </div>

**Other Ways to Support:**

- ⭐ Star the project on `GitHub <https://github.com/Asher-1/ACloudViewer>`_
- 🐛 Report bugs and suggest features
- 📝 Contribute code or documentation
- 💬 Help answer community questions
- 📢 Share ACloudViewer with others

Every contribution, no matter how small, helps make ACloudViewer better!

Contributing
------------

Help improve ACloudViewer:

🤝 **Contribute**

- **Code**: Submit pull requests
- **Documentation**: Improve docs
- **Bug Reports**: File issues
- **Feature Requests**: Propose ideas

See :doc:`/developer/contributing` for guidelines.

Response Times
--------------

**Community Channels:**

- GitHub Discussions: 1-3 days (community-driven)
- GitHub Issues: 1-7 days for bugs, longer for features
- Stack Overflow: Varies based on community activity

**Professional Support:**

- Email: 24-48 hours
- Enterprise: As per SLA agreement

Resources
---------

Code Examples
^^^^^^^^^^^^^

Browse example code in the repository:

- **Python**: ``examples/Python/``
- **C++**: ``examples/Cpp/``
- **Jupyter**: ``docs/jupyter/``

Videos & Tutorials
^^^^^^^^^^^^^^^^^^

Video tutorials and demos (coming soon):

- Getting started guides
- Feature walkthroughs
- Best practices

Research Papers
^^^^^^^^^^^^^^^

ACloudViewer is used in academic research. Cite as:

.. code-block:: bibtex

   @software{acloudviewer2024,
     title = {ACloudViewer: A Modern Library for 3D Data Processing},
     author = {ACloudViewer Development Team},
     year = {2024},
     url = {https://github.com/Asher-1/ACloudViewer}
   }

Community
---------

Stay Connected
^^^^^^^^^^^^^^

- 🌟 **Star on GitHub**: `ACloudViewer <https://github.com/Asher-1/ACloudViewer>`_
- 👁️ **Watch for Updates**: Get notified of new releases
- 🍴 **Fork**: Create your own version

Code of Conduct
^^^^^^^^^^^^^^^

We follow the `Contributor Covenant Code of Conduct <https://www.contributor-covenant.org/>`_.

All community members are expected to:

- Be respectful and inclusive
- Provide constructive feedback
- Help others when possible
- Follow community guidelines

Report Code of Conduct violations to: conduct@acloudviewer.org

Feature Requests
----------------

Want a new feature?

1. **Check existing requests**: Search GitHub Issues
2. **Discuss first**: Post in GitHub Discussions
3. **Submit formal request**: Create an issue with detailed description

**What to include:**

- Use case and motivation
- Proposed API or interface
- Examples of similar features in other libraries
- Willingness to contribute implementation

Bug Reports
-----------

Found a bug?

**Quality Bug Reports Include:**

1. **Clear title**: Brief description of the issue
2. **Environment**: OS, versions, hardware
3. **Minimal reproduction**: Smallest code to reproduce
4. **Expected vs actual**: What should happen vs what happened
5. **Screenshots/logs**: Visual evidence or error messages

**Template:**

.. code-block:: python

   import cloudViewer as cv3d
   
   # Minimal code to reproduce the bug
   pcd = cv3d.io.read_point_cloud("test.pcd")
   pcd.some_method()  # This crashes
   
   # Error message:
   # Segmentation fault...

Tips for Getting Help Faster
-----------------------------

1. **Search first**: Check docs, issues, discussions
2. **Be specific**: Provide context and details
3. **Minimal example**: Simplify your code to the essential problem
4. **Format code**: Use code blocks for readability
5. **Follow up**: Respond to clarifying questions
6. **Share solutions**: Post what worked for you

Quick Links
-----------

- 📖 **Documentation**: `https://asher-1.github.io/ACloudViewer/documentation/ <https://asher-1.github.io/ACloudViewer/documentation/>`_
- 💬 **Discussions**: `GitHub Discussions <https://github.com/Asher-1/ACloudViewer/discussions>`_
- 🐛 **Issues**: `GitHub Issues <https://github.com/Asher-1/ACloudViewer/issues>`_
- 📦 **Downloads**: `GitHub Releases <https://github.com/Asher-1/ACloudViewer/releases>`_
- 🤝 **Contributing**: :doc:`/developer/contributing`

Thank You
---------

Your feedback and questions help make ACloudViewer better for everyone. Thank you for being part of the community! 🎉
